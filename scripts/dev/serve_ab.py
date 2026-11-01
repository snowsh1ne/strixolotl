#!/usr/bin/env python3
# serve_ab.py NAME - fixed requests against a running server (port 8150): prose, code, code at temperature 0.7.
# Prints and appends to ~/.cache/xolotl/runs/dspark/serve-NAME.jsonl: wall s, usage, the engine's monitor timings.
import json, os, sys, time, urllib.request
NAME = sys.argv[1]
URL = "http://127.0.0.1:8150"
OUT = os.path.expanduser(f"~/.cache/xolotl/runs/dspark/serve-{NAME}.jsonl")
REQS = [
    ("prose", 0.0, "Explain in detail how a B-tree works: node layout, search, insertion with splits, deletion with "
                   "merges, and why databases prefer B+-trees. Use headings and a worked example."),
    ("code", 0.0, "Write a complete, production-quality Python module implementing a thread-safe LRU cache with "
                  "per-entry TTL, type hints and docstrings, followed by a pytest test suite covering eviction, "
                  "expiry and concurrency."),
    ("code-t07", 0.7, "Write a Rust command-line tool that recursively finds duplicate files by size and then by "
                      "BLAKE3 hash, with clap argument parsing, parallel hashing and a summary report."),
]

def call(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())

for tag, temp, text in REQS:
    t0 = time.time()
    r = call("/v1/chat/completions", {"model": "strixolotl-v4.1-flash", "max_tokens": 700, "temperature": temp,
                                      "messages": [{"role": "user", "content": text}]})
    wall = time.time() - t0
    rec = {"tag": tag, "temp": temp, "wall": round(wall, 1), "usage": r.get("usage")}
    try:
        rid = call("/api/requests")["requests"][0]["id"]
        rec["timings"] = call(f"/api/requests?id={rid}").get("timings")
    except Exception as e:
        rec["monitor_error"] = str(e)
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
