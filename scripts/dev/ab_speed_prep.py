# ab_speed_prep.py SESSION OUT [MAX_TOKENS [MIN_TOKENS]] - the warm speed bench's prompt (ab_speed.sh): a window of
# a held-out Claude Code session (not in any calibration corpus), rendered like build_corpus.py's session windows
# (V4.1 thinking mode, tools, effort 75) but cut BEFORE an assistant turn, so the prompt ends in <｜Assistant｜><think>
# like a pi request.  Picks, over windows starting at a user prompt in the session's middle half and rendering to
# MIN..MAX tokens (default 14000..16384), the cut whose real next assistant turn is longest (a long answer to give).
# Writes OUT/prompt.i32 and OUT/prompt.json.  Run in xolotl-quant (tokenizers).
import os, sys, json, numpy as np
from tokenizers import Tokenizer
SRC = os.path.expanduser("~/models/xolotl/DeepSeek-V4.1-Flash-1017")
os.environ["V41_ENCODING"] = f"{SRC}/encoding/encoding.py"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../xolotl/tools/v41"))
import build_corpus as B

path, out = sys.argv[1], sys.argv[2]
hi_t = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
lo_t = int(sys.argv[4]) if len(sys.argv) > 4 else 14000
tok = Tokenizer.from_file(f"{SRC}/tokenizer.json")
msgs, calls = B.session_messages(path)
tools = B.synth_tools(calls)
ntok = lambda s0, k: len(tok.encode(B.E.encode_messages([{"role": "system", "content": B.SYSTEM, "tools": tools}] + msgs[s0:k],
                                                         thinking_mode="thinking"), add_special_tokens=False).ids)
turn_len = lambda m: len(m["content"]) + len(m["reasoning_content"]) + sum(len(tc["function"]["arguments"]) for tc in m.get("tool_calls", []))
users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
cuts = [k for k in range(1, len(msgs)) if msgs[k]["role"] == "assistant" and msgs[k - 1]["role"] in ("user", "tool")]
best = None
for s0 in users[len(users) // 4: 3 * len(users) // 4]:
    ks = [k for k in cuts if k > s0]
    lo, hi, top = 0, len(ks) - 1, -1   # the last cut that still fits MAX tokens
    while lo <= hi:
        mid = (lo + hi) // 2
        if ntok(s0, ks[mid]) <= hi_t: top, lo = mid, mid + 1
        else: hi = mid - 1
    for j in range(top, -1, -1):
        n = ntok(s0, ks[j])
        if n < lo_t: break
        if best is None or turn_len(msgs[ks[j]]) > best[0]: best = (turn_len(msgs[ks[j]]), s0, ks[j], n)
assert best, "no window renders to MIN..MAX tokens"
L, s0, k, n = best
text = B.E.encode_messages([{"role": "system", "content": B.SYSTEM, "tools": tools}] + msgs[s0:k], thinking_mode="thinking")
ids = tok.encode(text, add_special_tokens=False).ids
assert text.endswith("<｜Assistant｜><think>"), text[-80:]
os.makedirs(out, exist_ok=True)
np.array(ids, np.int32).tofile(f"{out}/prompt.i32")
nxt = msgs[k]
json.dump({"session": os.path.basename(path), "start": s0, "cut": k, "messages": len(msgs), "tokens": len(ids),
           "next_turn_chars": L, "next_turn_tools": [tc["function"]["name"] for tc in nxt.get("tool_calls", [])],
           "last_user": next(m["content"][:300] for m in reversed(msgs[s0:k]) if m["role"] == "user")},
          open(f"{out}/prompt.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(ids)} tokens: messages {s0}..{k} of {len(msgs)}, next assistant turn {L} chars")
