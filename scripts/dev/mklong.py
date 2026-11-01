import importlib.util, sys, numpy as np
from tokenizers import Tokenizer
src, out = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("enc", f"{src}/encoding/encoding.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
tk = Tokenizer.from_file(f"{src}/tokenizer.json")
p = E.encode_messages([{"role": "user", "content": "Write a small but complete Python command-line tool that indexes a directory of Markdown notes into SQLite with full-text search, supports add/search/delete subcommands, and includes tests with pytest. Then explain the design decisions."}], thinking_mode="chat")
ids = tk.encode(p, add_special_tokens=False).ids
np.array(ids, np.int32).tofile(out); print(len(ids), "tokens")
