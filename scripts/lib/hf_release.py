#!/usr/bin/env python3
"""hf_release.py - the Strixolotl Hugging Face release: prepare, upload, fetch.  Called by scripts/hf_prepare,
scripts/hf_upload and scripts/download_model (run those, not this file).

  prepare  build the upload folder: small files, manifest.json, SHA256SUMS, the GGUF's byte-range parts
  upload   push the upload folder to an EXISTING repo, part by part, resumable (never creates or deletes repos)
  fetch    end-user download: parts -> verify -> append into one GGUF, small files, Engram shards, the .paths file

Repo layout (all paths relative to the repo root):
  README.md, LICENSE, manifest.json, SHA256SUMS
  <name>.gguf.part-NN-of-MM          plain byte ranges of the GGUF (`cat` order = NN), not llama.cpp gguf-split shards
  <name>.gguf.experts.json, rank-<B>.bin, tokenizer/*
  engram/model.safetensors.index.json, engram/config.json, engram/README.md
  engram/model-0004{7,8}-of-00048.safetensors   only with prepare --with-engram (a mirror of DeepSeek's files)
  dspark/<sidecar>.gguf                         with prepare --dspark-sidecar: the MTP draft stack for --dspark

The GGUF is only ever read (O_RDONLY, pread).  On btrfs the parts are reflinks of the GGUF's extents (FICLONERANGE):
no extra disk space and no copy.  Elsewhere (or --mode copy) `upload` cuts one part at a time, checks its sha256,
uploads it and deletes it.  Stdlib only, except huggingface_hub for upload / fetch.
"""
import argparse
import array
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time

CHUNK = 64 << 20
FICLONERANGE = 0x4020940D
FS_IOC_GETFLAGS, FS_IOC_SETFLAGS, FS_NOCOMP_FL = 0x80086601, 0x40086602, 0x400
BTRFS_MAGIC = "btrfs"
FORMAT = "strixolotl-release-1"

# The Engram tables the engine reads (ds41_engram.cpp Ds41Engram::open: layers.{1,14}.engram.embed.{weight,scale},
# found through model.safetensors.index.json).  They live in shards 47/48 of DeepSeek's checkpoint, unchanged by the
# quant.  Pinned to the revision this quant was made from; sizes and sha256 = the Hub's LFS oids (checked 2026-10-25
# against the API) = scripts/fetch_1017_fp8's .files.tsv on strixhalo.
ENGRAM_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ENGRAM_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ENGRAM_SHARDS = {
    "model-00047-of-00048.safetensors": (101535150936, "76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5"),
    "model-00048-of-00048.safetensors": (101537926640, "9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd"),
}

if os.environ.get("STRIXOLOTL_TEST_ENGRAM"):   # tests on dummy files only: {"name": [size, sha256], ...}
    ENGRAM_SHARDS = {k: tuple(v) for k, v in json.load(open(os.environ["STRIXOLOTL_TEST_ENGRAM"])).items()}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def die(msg):
    log("ERROR:", msg)
    sys.exit(1)


def gb(n):
    return f"{n / 1e9:.1f} GB"


def part_name(base, i, n):
    return f"{base}.part-{i + 1:02d}-of-{n:02d}"


def sha256_file(path, size=None, offset=0, progress=None):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        end = offset + (size if size is not None else os.fstat(fd).st_size - offset)
        off = offset
        t0 = time.time()
        while off < end:
            b = os.pread(fd, min(CHUNK, end - off), off)
            if not b:
                raise OSError(f"short read in {path}")
            h.update(b)
            off += len(b)
            if progress and (off - offset) % (8 << 30) < CHUNK:
                log(f"  {progress}: {gb(off - offset)} / {gb(end - offset)}, {(off - offset) / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s")
    finally:
        os.close(fd)
    return h.hexdigest()


def fs_type(path):
    # /proc/self/mountinfo: the longest mount point that prefixes the path
    path = os.path.realpath(path)
    best, typ = "", "unknown"
    with open("/proc/self/mountinfo") as f:
        for line in f:
            l, r = line.split(" - ", 1)
            mp = l.split()[4].replace("\\040", " ")
            if (path == mp or path.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best):
                best, typ = mp, r.split()[0]
    return typ


def ensure_nocompress(d):
    """btrfs: set chattr +m (FS_NOCOMP_FL) on a directory before any file is written into it.  Compressed extents turn
    the engine's O_DIRECT expert reads into buffered ones and make random Engram row reads decompress whole extents.
    Works without root and inside the toolbox (which has no chattr binary)."""
    os.makedirs(d, exist_ok=True)
    if fs_type(d) != BTRFS_MAGIC:
        return "not btrfs"
    fd = os.open(d, os.O_RDONLY)
    try:
        fl = array.array("l", [0])
        fcntl.ioctl(fd, FS_IOC_GETFLAGS, fl)
        if fl[0] & FS_NOCOMP_FL:
            return "btrfs, +m already set"
        fl[0] |= FS_NOCOMP_FL
        fcntl.ioctl(fd, FS_IOC_SETFLAGS, fl)
        fcntl.ioctl(fd, FS_IOC_GETFLAGS, fl)
        if not fl[0] & FS_NOCOMP_FL:
            die(f"could not set chattr +m on {d}: run `chattr +m {d}` on the host before downloading")
        return "btrfs, +m set now"
    finally:
        os.close(fd)


def free_bytes(d):
    st = os.statvfs(d)
    return st.f_bavail * st.f_frsize


def clone_range(src_fd, dst_path, offset, length, last):
    """Reflink [offset, offset+length) of src into a new file.  False if the filesystem cannot (then: copy mode)."""
    fd = os.open(dst_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.ioctl(fd, FICLONERANGE, struct.pack("qQQQ", src_fd, offset, 0 if last else length, 0))
        ok = os.fstat(fd).st_size == length
    except OSError:
        ok = False
    finally:
        os.close(fd)
    if not ok:
        os.remove(dst_path)
    return ok


# ---------------------------------------------------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------------------------------------------------
def engram_index(src_dir):
    """The minimal index: every layers.<L>.engram.* tensor -> its shard (the engine reads only the embed tables)."""
    p = os.path.join(src_dir, "model.safetensors.index.json")
    if not os.path.exists(p):
        die(f"no {p}: --engram-src must be the DeepSeek checkpoint dir (or a dir with its index + shards 47/48)")
    wm = json.load(open(p))["weight_map"]
    eng = {k: v for k, v in sorted(wm.items()) if ".engram." in k}
    need = [k for k in eng if ".engram.embed." in k]
    if not need or set(eng[k] for k in need) - set(ENGRAM_SHARDS):
        die(f"unexpected Engram layout in {p}: {sorted(set(eng.values()))}")
    return eng


def engram_sums(src_dir):
    """sha256 of shards 47/48: the pinned values, cross-checked with fetch_1017_fp8's .files.tsv when present."""
    tsv = os.path.join(src_dir, ".files.tsv")
    if os.path.exists(tsv):
        for line in open(tsv):
            f = line.rstrip("\n").split("\t")
            if len(f) == 3 and f[0] in ENGRAM_SHARDS and (int(f[1]), f[2]) != ENGRAM_SHARDS[f[0]]:
                die(f"{tsv}: {f[0]} differs from the pinned {ENGRAM_REPO}@{ENGRAM_REVISION[:12]} file")
    for fn, (size, _) in ENGRAM_SHARDS.items():
        p = os.path.join(src_dir, fn)
        if os.path.exists(p) and os.path.getsize(p) != size:
            die(f"{p} is {os.path.getsize(p)} bytes, expected {size}")
    return ENGRAM_SHARDS


ENGRAM_README = """# Engram tables

The engine reads DeepSeek-V4.1-Flash-1017's Engram n-gram tables at run time: four tensors,
`layers.1.engram.embed.{{weight,scale}}` and `layers.14.engram.embed.{{weight,scale}}` (F8_E4M3 rows of 256 bytes plus
F8_E8M0 scales, 202.8 GB together). They are unchanged DeepSeek weights, stored in two shards of the original
checkpoint. The quant does not touch them, so they are not repackaged here{mirror}.

| File | Size (bytes) | sha256 |
|---|---|---|
{rows}

Source: `{repo}` at revision `{rev}`.
`model.safetensors.index.json` in this folder maps the Engram tensors to those two shards; put it next to them and
pass the folder to the engine as `--engram DIR`. `scripts/download_model` in the Strixolotl repo does all of this.
"""


def cmd_prepare(a):
    gguf = os.path.realpath(a.gguf)
    if not os.path.isfile(gguf):
        die(f"no GGUF {gguf}")
    base = os.path.basename(gguf)
    out = os.path.realpath(a.out)
    if out == os.path.dirname(gguf) or out.startswith(os.path.dirname(gguf) + os.sep):
        die(f"the upload folder {out} must not be inside the GGUF's folder")
    st = os.stat(gguf)
    size = st.st_size
    part = int(a.part_gib * (1 << 30))
    if part % 4096:
        die("--part-gib must give a multiple of 4096 bytes")
    n = math.ceil(size / part)
    if n > 99:
        die(f"{n} parts: use a larger --part-gib (the names have two digits)")
    small = [(a.rank, os.path.basename(a.rank)), (a.experts_json, os.path.basename(a.experts_json))]
    for p, _ in small:
        if not os.path.isfile(p):
            die(f"missing {p}")
    tok = sorted(os.listdir(a.tokenizer))
    if not tok:
        die(f"empty tokenizer dir {a.tokenizer}")
    eng = engram_index(a.engram_src)
    sums = engram_sums(a.engram_src)
    if a.with_engram:
        for fn in sums:
            if not os.path.isfile(os.path.join(a.engram_src, fn)):
                die(f"--with-engram: missing {os.path.join(a.engram_src, fn)}")

    log(f"GGUF {gguf}: {size} bytes ({gb(size)}) -> {n} parts of {a.part_gib:g} GiB")
    probe = out
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    log(f"upload folder {out} (free {gb(free_bytes(probe))})")
    log(f"small files: {', '.join(d for _, d in small)}, tokenizer/ ({len(tok)} files), engram/ index + config, "
        f"README.md (from {a.card}), LICENSE (from {a.license})")
    if a.dspark_sidecar:
        log(f"DSpark sidecar: {a.dspark_sidecar} -> dspark/ ({gb(os.path.getsize(a.dspark_sidecar))}, its own commit)")
    log(f"Engram: {'MIRRORED (+' + gb(sum(s for s, _ in sums.values())) + ')' if a.with_engram else 'not mirrored: users fetch shards 47/48 from ' + ENGRAM_REPO}")
    if a.dry_run:
        for i in range(n):
            log(f"  {part_name(base, i, n)}  bytes {i * part}..{min(size, (i + 1) * part)}")
        log("dry run: nothing hashed or written")
        return

    os.makedirs(out, exist_ok=True)
    local = os.path.join(out, ".local")
    os.makedirs(local, exist_ok=True)
    state_p = os.path.join(local, "hashes.json")
    sid = {"path": gguf, "size": size, "mtime_ns": st.st_mtime_ns, "part_bytes": part}
    hs = json.load(open(state_p)) if os.path.exists(state_p) else {}
    if hs.get("source") != sid:
        log(f"hashing the GGUF once (read-only, {n} parts + the whole file; ~{size / 2.0e9 / 60:.0f}-{size / 1.0e9 / 60:.0f} min) ...")
        full = hashlib.sha256()
        parts = []
        fd = os.open(gguf, os.O_RDONLY)
        t0 = time.time()
        try:
            off = 0
            for i in range(n):
                h = hashlib.sha256()
                end = min(size, (i + 1) * part)
                while off < end:
                    b = os.pread(fd, min(CHUNK, end - off), off)
                    if not b:
                        die("short read")
                    h.update(b)
                    full.update(b)
                    off += len(b)
                parts.append(h.hexdigest())
                log(f"  {part_name(base, i, n)}: {gb(off)} / {gb(size)}, {off / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s")
        finally:
            os.close(fd)
        if os.stat(gguf).st_mtime_ns != st.st_mtime_ns:
            die("the GGUF changed while it was hashed")
        hs = {"source": sid, "full": full.hexdigest(), "parts": parts}
        json.dump(hs, open(state_p, "w"), indent=1)
    else:
        log("GGUF hashes: cached (same size, mtime and part size)")

    # small files (copied: they are tiny; the GGUF itself is never copied whole)
    files = []
    for p, dst in small:
        shutil.copyfile(p, os.path.join(out, dst))
        files.append(dst)
    if a.dspark_sidecar:   # the MTP draft stack (start_xolotl_dsv41 --dspark): a plain copy, ~6 GB
        sc = os.path.realpath(a.dspark_sidecar)
        if not os.path.isfile(sc):
            die(f"--dspark-sidecar: no {sc}")
        dst = os.path.join("dspark", os.path.basename(sc))
        os.makedirs(os.path.join(out, "dspark"), exist_ok=True)
        q = os.path.join(out, dst)
        if not (os.path.exists(q) and os.path.getsize(q) == os.path.getsize(sc) and os.path.getmtime(q) >= os.path.getmtime(sc)):
            log(f"copying the DSpark sidecar {sc} ({gb(os.path.getsize(sc))})")
            shutil.copyfile(sc, q + ".tmp")
            os.replace(q + ".tmp", q)
        files.append(dst)
    os.makedirs(os.path.join(out, "tokenizer"), exist_ok=True)
    for f in tok:
        shutil.copyfile(os.path.join(a.tokenizer, f), os.path.join(out, "tokenizer", f))
        files.append(f"tokenizer/{f}")
    ed = os.path.join(out, "engram")
    os.makedirs(ed, exist_ok=True)
    total = sum(s for s, _ in sums.values())
    with open(os.path.join(ed, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": eng}, f, indent=1)
    files.append("engram/model.safetensors.index.json")
    cfg = os.path.join(a.engram_src, "config.json")
    if os.path.isfile(cfg):
        shutil.copyfile(cfg, os.path.join(ed, "config.json"))
        files.append("engram/config.json")
    rows = "\n".join(f"| `{fn}` | {s} | `{h}` |" for fn, (s, h) in sums.items())
    with open(os.path.join(ed, "README.md"), "w") as f:
        f.write(ENGRAM_README.format(rows=rows, repo=ENGRAM_REPO, rev=ENGRAM_REVISION,
                                     mirror=" (this repo also carries a byte-identical mirror of the two shards)" if a.with_engram else ""))
    files.append("engram/README.md")
    shutil.copyfile(a.card, os.path.join(out, "README.md"))
    shutil.copyfile(a.license, os.path.join(out, "LICENSE"))

    # parts: reflinks where the filesystem can (no space, no copy), else cut later by `upload`
    mode = a.mode
    staged = 0
    if mode in ("auto", "reflink"):
        sfd = os.open(gguf, os.O_RDONLY)
        try:
            for i in range(n):
                nm = part_name(base, i, n)
                p = os.path.join(out, nm)
                ln = min(part, size - i * part)
                if os.path.exists(p) and os.path.getsize(p) == ln:
                    staged += 1
                    continue
                if not clone_range(sfd, p, i * part, ln, i == n - 1):
                    if mode == "reflink":
                        die(f"reflink failed for {nm} ({fs_type(out)}): use --mode copy")
                    log(f"no reflink on {fs_type(out)} between {os.path.dirname(gguf)} and {out}: parts are cut at upload time")
                    mode = "copy"
                    break
                staged += 1
        finally:
            os.close(sfd)
        if mode != "copy":
            mode = "reflink"
            log(f"{staged} parts reflinked into {out} (shared extents: no extra space)")
    if a.with_engram:
        for fn in sums:
            p = os.path.join(ed, fn)
            if not os.path.exists(p):
                sfd = os.open(os.path.join(a.engram_src, fn), os.O_RDONLY)
                try:
                    if not clone_range(sfd, p, 0, sums[fn][0], True):
                        die(f"--with-engram needs a reflink-capable filesystem (or copy {fn} into {ed} yourself)")
                finally:
                    os.close(sfd)
            files.append(f"engram/{fn}")

    def fsha(rel):
        return sha256_file(os.path.join(out, rel))

    man = {
        "format": FORMAT,
        "name": base[:-5] if base.endswith(".gguf") else base,
        "model": "DeepSeek-V4.1-Flash-1017",
        "quant": a.quant_desc,
        "served_model_id": "strixolotl-v4.1-flash",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gguf": {"name": base, "size": size, "sha256": hs["full"], "part_bytes": part,
                 "parts": [{"name": part_name(base, i, n), "offset": i * part, "size": min(part, size - i * part),
                            "sha256": hs["parts"][i]} for i in range(n)]},
        "files": [{"path": rel, "size": os.path.getsize(os.path.join(out, rel)),
                   "sha256": sums[rel[7:]][1] if rel[7:] in sums else fsha(rel)} for rel in files],
        "engram": {"repo": ENGRAM_REPO, "revision": ENGRAM_REVISION, "mirrored": bool(a.with_engram),
                   "index": "engram/model.safetensors.index.json",
                   "tensors_read": sorted(k for k in eng if ".engram.embed." in k),
                   "shards": [{"name": fn, "size": s, "sha256": h} for fn, (s, h) in sums.items()]},
    }
    if a.dspark_sidecar:
        man["dspark"] = {"sidecar": next(f for f in files if f.startswith("dspark/")),
                         "note": "MTP draft stack (mtp.0-2, experts IQ3_XXS) for start_xolotl_dsv41 --dspark"}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        for p in man["gguf"]["parts"]:
            f.write(f"{p['sha256']}  {p['name']}\n")
        f.write(f"{hs['full']}  {base}\n")
        for e in man["files"]:
            f.write(f"{e['sha256']}  {e['path']}\n")
    json.dump({"gguf": gguf, "mode": mode, "out": out}, open(os.path.join(local, "local.json"), "w"), indent=1)
    log(f"done: {out} (mode {mode}); next: scripts/hf_upload --dir {out} --repo <hf-user>/<repo> --dry-run")


# ---------------------------------------------------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------------------------------------------------
def cmd_upload(a):
    out = os.path.realpath(a.dir)
    man = json.load(open(os.path.join(out, "manifest.json")))
    loc = json.load(open(os.path.join(out, ".local", "local.json")))
    gguf = loc["gguf"]
    g = man["gguf"]
    # transport mode: the folder was copied to another machine (e.g. on an external SSD, see HF_RELEASE.md) without the
    # GGUF; then every part must be staged in it, each is checked against its sha256 before its upload, none is deleted
    transport = not (os.path.isfile(gguf) and os.path.getsize(gguf) == g["size"])
    if transport:
        missing = [p["name"] for p in g["parts"] if not (os.path.isfile(os.path.join(out, p["name"]))
                                                         and os.path.getsize(os.path.join(out, p["name"])) == p["size"])]
        if missing:
            die(f"{gguf} is not on this machine and {len(missing)} of {len(g['parts'])} parts are missing or short in {out} "
                f"(first: {missing[0]}): copy the whole folder, or run on the machine with the GGUF")
        log(f"transport mode: no GGUF here; uploading the {len(g['parts'])} staged parts from {out} "
            f"(each checked against its sha256 first; nothing is deleted)")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        die("huggingface_hub is missing: run this inside the xolotl-strix toolbox")
    api = HfApi()
    if not a.dry_run:
        try:
            who = api.whoami()
        except Exception as e:  # noqa: BLE001
            die(f"not logged in ({e.__class__.__name__}): run `hf auth login` with a write token first")
        log(f"logged in as {who.get('name')}")
    if not api.repo_exists(a.repo, repo_type="model"):
        die(f"{a.repo} does not exist (or is not visible to this token): create it on huggingface.co or with "
            f"`hf repos create {a.repo}` first; this script never creates repos")
    remote = {}
    for f in api.list_repo_tree(a.repo, recursive=True, revision=a.revision):
        if getattr(f, "size", None) is not None:
            remote[f.path] = f.size
    large = (".safetensors", ".gguf")   # their own commits (the Engram mirror, the DSpark sidecar)
    small = ["README.md", "LICENSE", "manifest.json", "SHA256SUMS"] + [e["path"] for e in man["files"]
                                                                      if not e["path"].endswith(large)]
    big = [(p["name"], p) for p in g["parts"]] + [(e["path"], None) for e in man["files"] if e["path"].endswith(large)]
    todo = [nm for nm, p in big if remote.get(nm) != (p["size"] if p else os.path.getsize(os.path.join(out, nm)))]
    log(f"{a.repo}: {len(remote)} files there; {len(todo)} of {len(big)} large files still to upload; mode {loc['mode']}")
    if a.dry_run:
        for nm, p in big:
            log(f"  {nm:50s} {'on the hub' if nm not in todo else 'TODO'}")
        log(f"small files (re-sent each run): {len(small)}")
        return

    # 1. the small files in one commit (README.md, manifest, SHA256SUMS, rank, experts.json, tokenizer/, engram/ meta)
    log("uploading the small files (one commit)")
    api.upload_folder(folder_path=out, repo_id=a.repo, revision=a.revision, allow_patterns=small,
                      commit_message=a.message or f"{man['name']}: card, manifest, checksums, small files")

    # 2. the parts, one per commit, each checked against the Hub's size afterwards
    part_re = re.compile(re.escape(g["name"]) + r"\.part-\d{2}-of-\d{2}(\.tmp)?$")
    for nm, p in big:
        if nm not in todo:
            continue
        path = os.path.join(out, nm)
        cut = False
        if p and not (os.path.exists(path) and os.path.getsize(path) == p["size"]):
            free = free_bytes(out)
            if free < p["size"] + (2 << 30):
                die(f"{nm}: needs {gb(p['size'])} free in {out}, {gb(free)} free")
            log(f"cutting {nm} from the GGUF ({gb(p['size'])})")
            tmp = path + ".tmp"
            h = hashlib.sha256()
            sfd = os.open(gguf, os.O_RDONLY)
            try:
                with open(tmp, "wb") as fo:
                    off, end = p["offset"], p["offset"] + p["size"]
                    while off < end:
                        b = os.pread(sfd, min(CHUNK, end - off), off)
                        if not b:
                            die("short read")
                        h.update(b)
                        fo.write(b)
                        off += len(b)
            finally:
                os.close(sfd)
            if h.hexdigest() != p["sha256"]:
                os.remove(tmp)
                die(f"{nm}: sha256 differs from the manifest - the GGUF changed since hf_prepare; stopping")
            os.replace(tmp, path)
            cut = True
        if transport and p:
            h = hashlib.sha256()
            with open(path, "rb") as fi:
                for b in iter(lambda: fi.read(CHUNK), b""):
                    h.update(b)
            if h.hexdigest() != p["sha256"]:
                die(f"{nm}: sha256 differs from the manifest (damaged copy?); copy it again from the source folder")
        t0 = time.time()
        size = os.path.getsize(path)
        log(f"uploading {nm} ({gb(size)})")
        api.upload_file(path_or_fileobj=path, path_in_repo=nm, repo_id=a.repo, revision=a.revision,
                        commit_message=f"add {nm}")
        dt = time.time() - t0
        info = api.get_paths_info(a.repo, [nm], revision=a.revision)
        got = info[0].size if info else None
        if got != size:
            die(f"{nm}: the Hub reports {got} bytes, expected {size}; stopping (re-run to retry)")
        log(f"{nm} done in {dt:.0f} s ({size / 1e6 / max(dt, 1e-3):.1f} MB/s)")
        if not transport and (cut or not a.keep_parts) and part_re.match(nm) and os.path.dirname(os.path.realpath(path)) == out:
            os.remove(path)   # a cut copy (or a reflink: frees nothing while the GGUF exists)
    log(f"ALL DONE: every part and small file of {man['name']} is on {a.repo}")


# ---------------------------------------------------------------------------------------------------------------------
# fetch (end users)
# ---------------------------------------------------------------------------------------------------------------------
def cmd_fetch(a):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("huggingface_hub is missing: run this inside the xolotl-strix toolbox (scripts/toolbox_create)")
    models = os.path.realpath(a.models)
    gsq = os.path.join(models, a.gsq_dir)
    for d in (models,):
        os.makedirs(d, exist_ok=True)
        t = fs_type(d)
        if t in ("nfs", "nfs4", "cifs", "smb2", "smb3", "exfat", "vfat", "fuseblk"):
            die(f"{d} is on {t}: put the models on a local ext4/btrfs/xfs NVMe (XOLOTL_MODELS=/path)")
    if not a.dry_run:
        log(f"{gsq}: {ensure_nocompress(gsq)}")   # before any file is written there
    meta = os.path.join(gsq, ".release")
    os.makedirs(meta, exist_ok=True)

    def get(repo, fn, dest_dir, rev):
        return hf_hub_download(repo_id=repo, filename=fn, revision=rev, local_dir=dest_dir)

    log(f"manifest from {a.repo}")
    mp = get(a.repo, "manifest.json", meta, a.revision)
    man = json.load(open(mp))
    if man.get("format") != FORMAT:
        die(f"unknown release format {man.get('format')} (this script knows {FORMAT})")
    g = man["gguf"]
    e = man["engram"]
    eng_dir = a.engram_dir or os.path.join(models, a.engram_name)
    eng_need = 0 if a.engram_dir else sum(s["size"] for s in e["shards"]
                                          if not (os.path.exists(os.path.join(eng_dir, s["name"]))
                                                  and os.path.getsize(os.path.join(eng_dir, s["name"])) == s["size"]))
    out = os.path.join(gsq, g["name"])
    done_flag = os.path.join(meta, "gguf.ok")
    st_p = os.path.join(meta, "assemble.json")
    have = 0
    if os.path.exists(done_flag) and os.path.exists(out) and os.path.getsize(out) == g["size"]:
        gg_need = 0
    else:
        st = json.load(open(st_p)) if os.path.exists(st_p) else {"done": 0}
        have = sum(p["size"] for p in g["parts"][:st["done"]])
        gg_need = g["size"] - have
    big_part = max(p["size"] for p in g["parts"])
    # the small files and the DSpark sidecar (5.9 GB) land next to the GGUF: count what is not there yet
    files_need = sum(f["size"] for f in man["files"] if not f["path"].startswith("engram/")
                     and not (os.path.exists(os.path.join(gsq, f["path"]))
                              and os.path.getsize(os.path.join(gsq, f["path"])) == f["size"]))
    need = gg_need + (big_part if gg_need else 0) + files_need + eng_need + (1 << 30)
    log(f"{man['name']}: GGUF {gb(g['size'])} in {len(g['parts'])} parts, Engram {gb(sum(s['size'] for s in e['shards']))} "
        f"({'from ' + eng_dir if a.engram_dir else ('mirror in ' + a.repo if a.engram_from == 'mirror' else e['repo'] + '@' + e['revision'][:12])})")
    log(f"to download: {gb(gg_need + files_need + eng_need)}; disk needed now: {gb(need)}; free in {models}: {gb(free_bytes(models))}")
    if a.dry_run:
        log(f"dry run: GGUF -> {out}, Engram -> {eng_dir}, paths -> {os.path.join(models, a.paths_name)}")
        return
    if free_bytes(models) < need:
        die(f"not enough space: need {gb(need)}, {gb(free_bytes(models))} free in {models}")
    if gg_need and os.path.exists(out):
        die(f"{out} already exists and was not written by this script: move it away (or use --gsq-dir)")

    # 1. the GGUF: one part at a time -> sha256 -> append at its offset -> delete the part
    if gg_need:
        st = json.load(open(st_p)) if os.path.exists(st_p) else {"done": 0, "name": g["name"], "sha256": g["sha256"]}
        if st.get("sha256", g["sha256"]) != g["sha256"]:
            die(f"{st_p} belongs to another release: move it and {out}.partial away")
        partial = out + ".partial"
        dl = os.path.join(gsq, ".dl")
        for i, p in enumerate(g["parts"]):
            if i < st["done"]:
                continue
            log(f"part {i + 1}/{len(g['parts'])}: {p['name']} ({gb(p['size'])})")
            pp = get(a.repo, p["name"], dl, a.revision)
            if os.path.getsize(pp) != p["size"]:
                os.remove(pp)
                die(f"{p['name']}: wrong size after download; re-run to retry")
            h = hashlib.sha256()
            with open(partial, "ab") as fo:
                fo.truncate(p["offset"])          # an interrupted append is cut back to this part's start
                fo.seek(p["offset"])
                with open(pp, "rb", buffering=0) as fi:
                    while True:
                        b = fi.read(CHUNK)
                        if not b:
                            break
                        h.update(b)
                        fo.write(b)
                fo.flush()
                os.fsync(fo.fileno())
            if h.hexdigest() != p["sha256"]:
                with open(partial, "ab") as fo:
                    fo.truncate(p["offset"])
                os.remove(pp)
                die(f"{p['name']}: sha256 mismatch (corrupt download); deleted it, re-run to fetch it again")
            os.remove(pp)
            st["done"] = i + 1
            json.dump(st, open(st_p + ".tmp", "w"))
            os.replace(st_p + ".tmp", st_p)
        shutil.rmtree(dl, ignore_errors=True)
        log(f"verifying the whole GGUF ({gb(g['size'])}, a few minutes)")
        if sha256_file(partial, progress="sha256") != g["sha256"]:
            die(f"{partial}: the joined file's sha256 does not match the manifest; delete it and {st_p}, then re-run")
        os.replace(partial, out)
        open(done_flag, "w").write(g["sha256"] + "\n")
        os.remove(st_p)
    log(f"GGUF ok: {out}")

    # 2. small files into the GSQ dir (rank, experts.json, tokenizer/), checked against the manifest
    for f in man["files"]:
        if f["path"].startswith("engram/"):
            continue
        p = get(a.repo, f["path"], gsq, a.revision)
        if os.path.getsize(p) != f["size"] or sha256_file(p) != f["sha256"]:
            die(f"{f['path']}: checksum mismatch")
    for extra in ("README.md", "LICENSE", "SHA256SUMS"):
        get(a.repo, extra, meta, a.revision)
    log("small files ok")

    # 3. Engram: DeepSeek's two shards (or the mirror, or an existing dir) + the minimal index
    if a.engram_dir:
        for s in e["shards"]:
            p = os.path.join(eng_dir, s["name"])
            if not os.path.isfile(p) or os.path.getsize(p) != s["size"]:
                die(f"--engram-dir: {p} missing or not {s['size']} bytes")
            if a.verify_engram:
                log(f"sha256 {s['name']} ({gb(s['size'])})")
                if sha256_file(p, progress=s["name"]) != s["sha256"]:
                    die(f"{p}: sha256 mismatch")
        if not os.path.exists(os.path.join(eng_dir, "model.safetensors.index.json")):
            die(f"--engram-dir: no model.safetensors.index.json in {eng_dir}")
    else:
        log(f"{eng_dir}: {ensure_nocompress(eng_dir)}")
        for s in e["shards"]:
            p = os.path.join(eng_dir, s["name"])
            ok = os.path.join(eng_dir, ".release", s["name"] + ".ok")
            if os.path.exists(ok) and os.path.exists(p) and os.path.getsize(p) == s["size"]:
                continue
            log(f"Engram {s['name']} ({gb(s['size'])})")
            if a.engram_from == "mirror":
                q = get(a.repo, "engram/" + s["name"], os.path.join(eng_dir, ".dl"), a.revision)
                os.replace(q, p)
            else:
                get(e["repo"], s["name"], eng_dir, e["revision"])
            if os.path.getsize(p) != s["size"] or sha256_file(p, progress=s["name"]) != s["sha256"]:
                os.remove(p)
                die(f"{s['name']}: checksum mismatch; deleted, re-run to fetch it again")
            os.makedirs(os.path.dirname(ok), exist_ok=True)
            open(ok, "w").write(s["sha256"] + "\n")
        shutil.rmtree(os.path.join(eng_dir, ".dl"), ignore_errors=True)
        for fn in ("model.safetensors.index.json", "config.json", "README.md"):
            q = get(a.repo, "engram/" + fn, os.path.join(eng_dir, ".release"), a.revision)
            shutil.copyfile(q, os.path.join(eng_dir, fn))
    log(f"Engram ok: {eng_dir}")

    # 4. the paths file that scripts/start_xolotl_dsv41 and scripts/verify_setup read
    rank = next(f["path"] for f in man["files"] if re.match(r"rank.*\.bin$", f["path"]))
    pf = os.path.join(models, a.paths_name)
    body = (f"XOLOTL_DSV41_NAME={a.paths_name[:-6]}\n"
            f"XOLOTL_DSV41_REPO=hf:{a.repo}{'@' + a.revision if a.revision else ''}\n"
            f'XOLOTL_DSV41_GGUF="{out}"\n'
            f'XOLOTL_DSV41_RANK="{os.path.join(gsq, rank)}"\n'
            f'XOLOTL_DSV41_TOKENIZER="{os.path.join(gsq, "tokenizer")}"\n'
            f'XOLOTL_DSV41_ENGRAM="{eng_dir}"\n'
            f'XOLOTL_DSV41_DSPARK="{os.path.join(gsq, man["dspark"]["sidecar"]) if man.get("dspark") else ""}"\n')
    old = open(pf).read() if os.path.exists(pf) else None
    if old is not None and old.split("\n", 1)[-1] != body:
        bak = pf + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
        os.replace(pf, bak)
        log(f"kept the old paths file as {bak}")
    if old is None or old.split("\n", 1)[-1] != body:
        with open(pf, "w") as f:
            f.write(f"# written by scripts/download_model on {datetime.datetime.now().isoformat(timespec='seconds')}: "
                    f"{man['name']} from {a.repo}\n" + body)
    log(f"done: {pf}")
    sys.stdout.write(open(pf).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--gguf", required=True)
    p.add_argument("--rank", required=True)
    p.add_argument("--experts-json", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--engram-src", required=True)
    p.add_argument("--card", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part-gib", type=float, default=10)
    p.add_argument("--mode", choices=("auto", "reflink", "copy"), default="auto")
    p.add_argument("--with-engram", action="store_true")
    p.add_argument("--dspark-sidecar", default=None)
    p.add_argument("--quant-desc", default="")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("upload")
    p.add_argument("--dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--message", default=None)
    p.add_argument("--keep-parts", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sp.add_parser("fetch")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--models", required=True)
    p.add_argument("--gsq-dir", default="V41-1017-GSQ")
    p.add_argument("--engram-name", default="V41-1017-Engram")
    p.add_argument("--engram-from", choices=("deepseek", "mirror"), default="deepseek")
    p.add_argument("--engram-dir", default=None)
    p.add_argument("--verify-engram", action="store_true")
    p.add_argument("--paths-name", default="V41-1017.paths")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "upload": cmd_upload, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
