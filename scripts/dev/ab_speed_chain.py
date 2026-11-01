# ab_speed_chain.py SESSION OUT PREV.i32 GEN.ids STEP - the next prompt of the warm bench's forced-stream chain: when
# the generation from PREV.i32 (ab_speed_prep.py's prompt.i32, or the previous chain prompt) ends its turn before 3000
# tokens, continue the conversation: PREV + the generated turn (GEN.ids, ending in <｜end▁of▁sentence｜>) + the
# session's real message(s) after its real assistant turn number STEP after the cut (the tool results / user reply,
# rendered as the encoder does, ending in <｜Assistant｜><think>).  Writes OUT/chain<STEP>.i32.  Run in xolotl-quant.
import os, sys, json, numpy as np
from tokenizers import Tokenizer
SRC = os.path.expanduser("~/models/xolotl/DeepSeek-V4.1-Flash-1017")
os.environ["V41_ENCODING"] = f"{SRC}/encoding/encoding.py"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../xolotl/tools/v41"))
import build_corpus as B

path, out, prev, gen, step = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
meta = json.load(open(f"{out}/prompt.json"))
tok = Tokenizer.from_file(f"{SRC}/tokenizer.json")
msgs, calls = B.session_messages(path)
tools = B.synth_tools(calls)
enc = lambda k: B.E.encode_messages([{"role": "system", "content": B.SYSTEM, "tools": tools}] + msgs[meta["start"]:k],
                                    thinking_mode="thinking")
# the real assistant turn number `step` after the cut (1 = the one at the cut), then the messages up to the next one;
# only assistant messages a tool result / user reply follows count (Claude Code logs some turns as several messages)
asst = [k for k in range(meta["cut"], len(msgs) - 1) if msgs[k]["role"] == "assistant" and msgs[k + 1]["role"] != "assistant"]
a = asst[step - 1]
b = next(k for k in range(a + 1, len(msgs)) if msgs[k]["role"] == "assistant")
t1, t2 = enc(a + 1), enc(b)
assert t2.startswith(t1) and t2.endswith("<｜Assistant｜><think>"), (t1[-120:], t2[len(t1):][:200])
suffix = tok.encode(t2[len(t1):], add_special_tokens=False).ids
g = [int(x) for x in open(gen) if x.strip().isdigit()]
assert g and g[-1] == 1, "the generation did not end its turn"
ids = np.concatenate([np.fromfile(prev, "<i4"), np.array(g + suffix, np.int32)])
ids.astype("<i4").tofile(f"{out}/chain{step}.i32")
print(f"chain{step}: {len(ids)} tokens = prev + {len(g)} generated + {len(suffix)} real ({b - a - 1} messages: "
      f"{[m['role'] for m in msgs[a + 1:b]]})")
