# Releasing GSQ-217 on Hugging Face

Step-by-step for publishing the Strixolotl quant so that users can run it without quantizing. You run every step that
touches Hugging Face. The scripts never create, rename or delete repos.

What gets uploaded (~243 GB):
- the GGUF as 23 byte-range parts of 10 GiB (`V41-1017-GSQ-217.gguf.part-NN-of-23`), not llama.cpp `gguf-split`
  shards
- `README.md` (from `hf/MODEL_CARD.md`), `LICENSE` (DeepSeek's MIT text), `manifest.json`, `SHA256SUMS`
- `rank-217.bin`, `V41-1017-GSQ-217.gguf.experts.json`, `tokenizer/`
- `engram/`: a minimal `model.safetensors.index.json`, `config.json` and a README with the pinned source

The Engram tables (203.1 GB) are **not** uploaded by default. `scripts/download_model` fetches the two shards that
hold them from `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, and
checks their sha256 (the same values as the Hub's LFS ids for that revision, checked 2026-10-25).

## Hugging Face limits that matter here

From <https://huggingface.co/docs/hub/storage-limits> (read 2026-10-25):
- **File size:** hard limit 500 GB per file; recommended under 200 GB per file ("split data into chunked files").
  Huge files are not cached by the CDN. The GGUF (243.1 GB) is under the hard limit but over the recommendation,
  so it goes up in 10 GiB parts: small resume units, and only one part of local disk in copy mode.
- **Storage:** free accounts get "best-effort" public storage and 100 GB of private storage. PRO gets up to 10 TB of
  public storage. A 243 GB **private** repo does not fit a free account's 100 GB. Decided (2026-10-25): PRO (1 TB
  private included, $18/TB/month above), upload private, polish, then switch to public in the repo settings.
  While private only you (or members of an org that owns it) can download; a replaced part keeps counting
  against storage until the history is squashed.
- **Commits:** under 100 files per commit recommended; `hf upload` splits large folders by itself.

From <https://huggingface.co/docs/huggingface_hub/guides/upload> (huggingface_hub 2.3.0 docs):
- Uploads go through `hf_xet` (installed with `huggingface_hub`, also in the `xolotl-strix` toolbox). A re-run of
  an interrupted upload skips committed files and already-uploaded chunks.
- `HF_XET_HIGH_PERFORMANCE=1` uses all bandwidth and cores. Upstream Xolotl saw timeouts on a home uplink and used
  `HF_XET_FIXED_UPLOAD_CONCURRENCY=3` (`xolotl/tools/v41/hf/upload-start.sh`).
- The CLI is `hf` (`hf auth login`, `hf repos create`, `hf upload`, `hf download`). The host's old
  `huggingface-cli` 0.30.2 is not used; run everything below inside `toolbox enter xolotl-strix`.

## Steps

### 1. Before you start

- Wait until nothing else needs the NVMe: `hf_prepare` reads the whole 243 GB GGUF once (~2-4 min), and the upload
  reads it again. Don't run either while a long test or a speed measurement is running.
- Disk: on strixhalo (btrfs) the parts are reflinks of the GGUF: no extra space. On other filesystems the upload
  cuts one part at a time (~11 GB free needed).

### 2. Account, token, repo

1. Create or log in to the Hugging Face account that will own the repo (the release: `Snowsh1ne`).
2. Settings -> Access Tokens: create a token with **write** access (a fine-grained token limited to the new repo is
   enough).
3. Create the model repo, on the website (New -> Model) or from the toolbox:
   ```bash
   toolbox enter xolotl-strix
   hf auth login                                   # paste the token (stored in your home dir, which the toolbox shares)
   hf auth whoami
   hf repos create <hf-user>/<repo>                # public; add --private only with PRO (see the limits above)
   ```
   The release repo: `Snowsh1ne/DeepSeek-V4.1-Flash-1017-GSQ-217-Strixolotl` (private until the GitHub release).

### 3. Fill the model card

Edit `hf/MODEL_CARD.md` (it becomes the repo's `README.md`). Fill or remove every `TODO`:

```bash
grep -n TODO hf/MODEL_CARD.md
```

- the Strixolotl GitHub URL (twice) and the HF repo id in the run instructions
- the long-context line (the main session fills it after the 1M test, or delete it)
- links for strixite and halogen in the credits

`hf_prepare` and `hf_upload` warn while TODOs are left. A card edit after the upload only needs `hf_prepare` (cached
hashes, seconds) and `hf_upload` again: the small files are re-sent on every run.

### 4. Build the upload folder

```bash
cd ~/projects/Xolotl_StrixHalo
scripts/hf_prepare --dry-run      # the plan: 23 parts, the small files, the free space
scripts/hf_prepare                # hashes the GGUF once, writes ~/.cache/xolotl/hf-release/V41-1017-GSQ-217/
```

Check the result:
```bash
D=~/.cache/xolotl/hf-release/V41-1017-GSQ-217
ls -la $D $D/engram
head -3 $D/SHA256SUMS; grep -c part- $D/SHA256SUMS          # 23 parts
python3 -I -m json.tool $D/manifest.json | head -30
```

Optional: `--with-engram` also puts DeepSeek's two Engram shards (2 x 101.5 GB, reflinked) into `engram/` for the
upload, as a mirror in case DeepSeek's repo changes. That doubles the upload to ~446 GB; `download_model
--engram-from mirror` then uses it.

### 5. Upload

```bash
scripts/hf_upload --repo <hf-user>/<repo> --dry-run      # lists what is there and what is still to do
setsid nohup scripts/hf_upload --repo <hf-user>/<repo> > ~/.cache/xolotl/hf-upload.log 2>&1 < /dev/null &
tail -f ~/.cache/xolotl/hf-upload.log
```

(inside the toolbox; or from the host with `toolbox run -c xolotl-strix scripts/hf_upload ...`.)

- First the small files in one commit, then one commit per part. After each part the script asks the Hub for the
  file's size and stops on a mismatch.
- Resumable: run the same command again after any error; parts already on the Hub are skipped.
- **If a part fails with `invalid shard failed to verify cas info for xorb ...; xorb not found`** (seen 2026-10-28,
  twice on part 1, hf_xet 1.7.0): the server's global dedupe points at a block it does not have. Move
  `~/.cache/huggingface/xet/<endpoint>/` aside and re-run with `HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=false`
  (no loss: a new model has nothing to dedupe against).
- Time: the upload is limited by your uplink. 243 GB take ~5.4 h at 100 Mbit/s and ~33 min at 1 Gbit/s, plus overhead.
- **Plan (2026-10-27): upload directly from strixhalo** over the home uplink (40 Mbit/s = 5 MB/s): ~13.5 h at line
  rate, realistically 15-16 h, so start it in the evening and let it run into the next day (resumable). The GGUF now
  sits on the 990 Pro's ext4, so `hf_prepare` uses copy mode: each 10 GiB part is cut just before its upload and
  deleted after it (~10 GiB free needed). The upload fills the uplink: remote use of the server over Tailscale gets
  laggier while it runs.

### 5b. Alternative, not planned: upload from another machine (external SSD)

Only if the direct upload is not an option: carry the upload folder to a faster connection. What travels: the folder from
step 4 (23 parts, 243.1 GB, plus ~6 MB of small files) and the three upload scripts. The GGUF itself stays home.

1. On strixhalo, after step 4: plug in an external SSD with at least 256 GB free, formatted ext4 (Linux) or exFAT
   (any OS; not FAT32, which stops at 4 GB per file). Copy the folder and the scripts (the parts are reflinks at home,
   so this reads 243 GB from the NVMe; ~5-10 min over USB 10 Gbit/s, longer over slower links):

   ```bash
   S=/run/media/$USER/<ssd>                    # the SSD's mount point
   rsync -a --info=progress2 ~/.cache/xolotl/hf-release/V41-1017-GSQ-217/ $S/strixolotl-upload/
   mkdir -p $S/strixolotl-scripts/lib
   cp scripts/hf_upload $S/strixolotl-scripts/ && cp scripts/lib/common.sh scripts/lib/hf_release.py $S/strixolotl-scripts/lib/
   (cd $S/strixolotl-upload && sha256sum -c --ignore-missing SHA256SUMS)   # checks the copy before you leave
   ```

2. On the other machine (Linux; macOS should work too; Windows: inside WSL): install the Hub client, log in, upload.

   ```bash
   python3 -m venv ~/hfvenv && ~/hfvenv/bin/pip install -U "huggingface_hub[hf_xet]"
   export PATH=~/hfvenv/bin:$PATH
   hf auth login                                # a write token
   S=/media/<ssd>                               # wherever the SSD is mounted there
   bash $S/strixolotl-scripts/hf_upload --dir $S/strixolotl-upload --repo <hf-user>/<repo> --dry-run
   bash $S/strixolotl-scripts/hf_upload --dir $S/strixolotl-upload --repo <hf-user>/<repo>
   ```

   Without the GGUF on that machine, `hf_upload` runs in transport mode: it checks every part against its sha256
   before uploading it (a damaged copy stops the run with the part's name) and deletes nothing. Re-run the same
   command after any interruption; parts already on the Hub are skipped. `HF_XET_HIGH_PERFORMANCE=1` uses the whole
   connection.

3. Back home, delete the upload folder (step 7) once `scripts/download_model` has verified the repo (step 6).

### 6. Check the repo

```bash
hf download <hf-user>/<repo> manifest.json SHA256SUMS --local-dir /tmp/strixolotl-check
diff /tmp/strixolotl-check/SHA256SUMS ~/.cache/xolotl/hf-release/V41-1017-GSQ-217/SHA256SUMS && echo same
scripts/download_model --repo <hf-user>/<repo> --dry-run     # reads the manifest, prints the plan
```

- On the website: 23 parts, sizes 10.7 GB each (the last 6.9 GB), the model card renders, the license shows "mit",
  "Quantized from deepseek-ai/DeepSeek-V4.1-Flash-1017" appears.
- A full download test needs ~460 GB free and must not overwrite the live files. strixhalo has ~238 GB free, so do
  it on another box, or point it at another disk: `XOLOTL_MODELS=/other/disk scripts/download_model --repo ...`.
  Then `XOLOTL_MODELS=/other/disk scripts/verify_setup` (host; the server stopped) must PASS with 32/32 tokens.
- Spot check without the full download: fetch one part and compare its sha256 with `SHA256SUMS`:
  ```bash
  hf download <hf-user>/<repo> V41-1017-GSQ-217.gguf.part-23-of-23 --local-dir /tmp/strixolotl-check
  (cd /tmp/strixolotl-check && grep part-23 SHA256SUMS | sha256sum -c -)
  ```

### 7. Publish

- If the repo is private: Settings -> make it public.
- Put the repo id into `README.md` and `QUICKSTART.md` (the `<hf-user>/<repo>` placeholders) and, if you want, as the
  default of `scripts/download_model --repo`.
- Delete the upload folder when done: `rm -rf ~/.cache/xolotl/hf-release/V41-1017-GSQ-217` (reflinked parts are
  deleted after upload anyway; they share the GGUF's extents).

## License: what the repo must carry

- DeepSeek-V4.1-Flash-1017 is MIT-licensed, weights included (model card: "This repository and the model weights are
  licensed under the MIT License"; `LICENSE` in the checkpoint: "Copyright (c) 2023 DeepSeek").
- MIT allows redistribution of derived weights. The condition is that the copyright notice and the permission notice
  are included in all copies or substantial portions. `hf_prepare` copies DeepSeek's `LICENSE` into the repo, and the
  card's front matter says `license: mit` with `base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017`.
- The Engram tables are unmodified DeepSeek weights; when mirrored (`--with-engram`), the same LICENSE covers them.
- The engine code (Xolotl/Strata, MIT) is not part of the model repo.

## Files

| File | What |
|---|---|
| `scripts/hf_prepare` | builds the upload folder; `--dry-run`, `--part-gib`, `--mode auto|reflink|copy`, `--with-engram` |
| `scripts/hf_upload` | uploads it to an existing repo; `--dry-run`, resumable |
| `scripts/download_model` | the user side: download, verify, assemble, write the `.paths` file |
| `scripts/lib/hf_release.py` | the code behind all three (stdlib + huggingface_hub) |
| `hf/MODEL_CARD.md` | the model card |
