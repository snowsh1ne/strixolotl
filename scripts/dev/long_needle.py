# long_needle.py DIR [URL] - long-mode test against a running server (start_xolotl_dsv41 --ctx 1048576): sends
# DIR/turns.json (long_needle_prep.py) as ONE growing conversation (each request = the whole history + the next turn,
# so the server's prompt reuse prefills every token once), thinking off, greedy.  Per turn: wall time, prompt /
# cached tokens and the engine timings (the request monitor), min MemAvailable while it ran, the answer and the pass
# keys recalled.  Appends DIR/results.jsonl (one line per turn; resumable: finished turns are replayed from it).
# Host python (stdlib only).
import sys, os, json, time, threading, re, urllib.request
D = sys.argv[1]; URL = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8150").rstrip("/")
turns = json.load(open(f"{D}/turns.json"))
done = [json.loads(l) for l in open(f"{D}/results.jsonl")] if os.path.exists(f"{D}/results.jsonl") else []
def memavail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) // 1024
def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=None) as f: return json.load(f)
def get(path):
    with urllib.request.urlopen(URL + path, timeout=60) as f: return json.load(f)
msgs = []
for i, t in enumerate(turns):
    msgs.append({"role": "user", "content": t["content"]})
    if i < len(done):   # replay a finished turn
        msgs.append({"role": "assistant", "content": done[i]["answer"]}); continue
    low, stop = [memavail()], threading.Event()
    def watch():
        while not stop.wait(1): low.append(memavail())
    th = threading.Thread(target=watch, daemon=True); th.start()
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": "x", "messages": msgs, "temperature": 0, "reasoning_effort": "none",
                                     "max_tokens": 1500 if t.get("decode_test") else 300})
    wall = time.time() - t0; stop.set(); th.join()
    ans = r["choices"][0]["message"].get("content") or ""
    got = {int(a): int(b) for a, b in re.findall(r"#\s*(\d+)\s*:\s*(\d+)", ans)}
    want = {k["n"]: k["value"] for k in t["all_keys"]}
    rec = {"turn": i, "target": t.get("target"), "wall_s": round(wall, 1), "usage": r.get("usage"),
           "min_memavail_mib": min(low), "keys_right": sorted(n for n, v in want.items() if got.get(n) == v),
           "keys_wanted": sorted(want), "answer": ans}
    try:   # the engine's own timings for this request (the monitor keeps the last 100)
        rid = get("/api/requests")["requests"][0]["id"]; rec["monitor"] = get(f"/api/requests?id={rid}").get("timings")
    except Exception as e:
        rec["monitor_error"] = str(e)
    open(f"{D}/results.jsonl", "a").write(json.dumps(rec) + "\n")
    print(f"turn {i} (target {t.get('target')}): {wall:.0f} s, usage {r.get('usage')}, keys "
          f"{len(rec['keys_right'])}/{len(want) if not t.get('decode_test') else 0}, min MemAvailable "
          f"{min(low) / 1024:.1f} GiB", flush=True)
    msgs.append({"role": "assistant", "content": ans})
