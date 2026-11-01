import importlib.util, json, sys, numpy as np
from tokenizers import Tokenizer
src, work, out = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("enc", f"{src}/encoding/encoding.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
tk = Tokenizer.from_file(f"{src}/tokenizer.json")
p = E.encode_messages([{"role": "user", "content": "Write a Python function that reverses a linked list, then explain its time complexity in two sentences."}], thinking_mode="chat")
ids = tk.encode(p, add_special_tokens=False).ids
np.array(ids, np.int32).tofile(f"{out}/chat.i32"); print("chat prompt", len(ids), "tokens:", repr(p[:120]))
meta = json.load(open(f"{work}/meta.json")); tok = np.load(f"{work}/tokens.npy")
for dom in ("code", "agent"):
    i = next(k for k, (d, e) in enumerate(zip(meta["domains"], meta["is_eval"])) if d == dom and e)
    tok[i, :4096].astype(np.int32).tofile(f"{out}/{dom}4k.i32"); print(dom, "eval seq", i)
