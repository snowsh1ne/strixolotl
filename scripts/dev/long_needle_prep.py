# long_needle_prep.py OUT [SRC_DIR] - inputs for long_needle.py (long mode, ROADMAP "Long-context mode"): one conversation that
# grows to ~1M tokens in stages (64K, 128K, 256K, 512K, ~1M of prompt).  Each stage's user message = filler (source
# files of SRC_DIR, default the llama.cpp clone, in sorted order, "// ===== file: <path> =====" headers) with two
# "PASS KEY #n = NNNNN" lines at 25% / 75% of that chunk, then a question asking for EVERY key so far - retrieval
# from all depths at each length.  Chunk sizes come from the 1017 tokenizer (the chat rendering adds a few tokens
# per turn; long_needle.py reports the server's real prompt sizes).  Writes OUT/turns.json.  Run in xolotl-quant.
import os, sys, json, random
from tokenizers import Tokenizer
OUT = sys.argv[1]
SRC = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/xolotl/llama.cpp")
tok = Tokenizer.from_file(os.path.expanduser("~/models/xolotl/DeepSeek-V4.1-Flash-1017/tokenizer.json"))
STAGES = [65536, 131072, 262144, 524288, 1040000]   # prompt size the stage's question is asked at
ANSWER = 400                                          # room per turn for the question + the answer in the history
Q = ("Ignore the code and documents above, they are filler. Somewhere in this conversation there are lines of the form "
     "'PASS KEY #<n> = <number>'. List every such pass key that appeared so far, one per line, exactly as "
     "'#<n>: <number>', in order, and nothing else.")
files = sorted(os.path.join(d, f) for d, _, fs in os.walk(SRC) if "/.git" not in d
               for f in fs if f.endswith((".cpp", ".h", ".c", ".md", ".py")))
def blocks():   # file texts in order, each with its header; endless (wraps around)
    while True:
        for f in files:
            try:
                t = open(f, errors="ignore").read()
            except OSError:
                continue
            if t.strip():
                yield f"// ===== file: {os.path.relpath(f, SRC)} =====\n{t}\n"
gen, rng = blocks(), random.Random(1017)
turns, keys, total, kn = [], [], 0, 0
for target in STAGES:
    need = target - total - ANSWER - len(tok.encode(Q).ids)
    parts, n = [], 0
    while n < need:   # whole files until the chunk is full (the last one cut at a line)
        b = next(gen); bn = len(tok.encode(b).ids)
        if n + bn > need:
            lines, keep = b.split("\n"), []
            for ln in lines:
                ln_n = len(tok.encode(ln + "\n").ids)
                if n + ln_n > need: break
                keep.append(ln); n += ln_n
            parts.append("\n".join(keep) + "\n"); break
        parts.append(b); n += bn
    stage_keys = []
    for frac in (0.25, 0.75):   # a key line between two files near that depth of the chunk
        kn += 1; val = rng.randint(10000, 99999)
        at = max(1, min(len(parts) - 1, int(len(parts) * frac)))
        parts.insert(at, f"\nPASS KEY #{kn} = {val}\n\n")
        stage_keys.append({"n": kn, "value": val, "depth_in_chunk": frac})
    keys += stage_keys
    text = "".join(parts) + "\n" + Q
    turns.append({"target": target, "chunk_tokens": n, "keys": stage_keys, "all_keys": list(keys), "content": text})
    total = target
    print(f"stage {target}: chunk {n} tokens, keys {[k['n'] for k in stage_keys]}")
# last turn: a long answer at ~1M for the decode speed at depth
turns.append({"target": None, "keys": [], "all_keys": list(keys), "decode_test": True,
              "content": "Now, without listing pass keys, explain in detail (about 800 words) what the first source file in "
                         "this conversation does and how its main functions work."})
os.makedirs(OUT, exist_ok=True)
json.dump(turns, open(f"{OUT}/turns.json", "w"))
print(f"{len(turns)} turns, {len(keys)} keys -> {OUT}/turns.json")
