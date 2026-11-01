# DSpark serve: prompt reuse after a cancel (analysis + fix)

Status 2026-10-28: analysis done by reading the code, the patch is written, **nothing built or run** (the server was
running). The patch is `docs/CANCEL_REUSE.patch`. Apply it from the repo root with `patch -p1 < docs/CANCEL_REUSE.patch`.
It applies cleanly: dry-run against the current tree, plus a real apply to a pristine copy in
`~/.cache/xolotl/runs/cancel/t`. It has not been compiled.

Line numbers below are for the **unpatched** sources.

## 1. Root cause

The hypothesis is right. Here is the exact mechanism.

**Engine, DSpark serve loop** (`xolotl/tools/v41/engine/ds41_generate.cpp`):

- 1768 `while (emit(t))`, then 1769 `if (stop_pending()) { finish = "cancel"; break; }`. STOP is only checked here,
  right after the loop-head `emit(t)`. That `t` was printed but is **not** fed and **not** in `hist` (1847: "the last
  emitted token is not fed").
- 1815 `pass_v` verifies `[t, d0..dk-1]`. Then 1835 `hist.push_back(t)` and 1837
  `for (...) { hist.push_back(drafts[i]); more = emit(drafts[i]); }`. The accepted drafts are pushed and printed back to
  back, with no STOP check in between, and then the next loop-head `emit(tn)` prints one more token.
- So at a cancel, `hist` = prompt + every emitted token except the last. The engine has emitted up to 1 + `acc` tokens
  past the one the front end cut at. At depth 4 that is up to 4 extra in `hist`; with `STRATA_SERVE_LOOKUP`, up to 14.

**Front end** (`xolotl/serve/server.py`):

- `Service._run` (4379-4450): `for t in gen:` counts the reasoning tokens. At 4446-4450, `thought >= budget` at a clean
  point sets `wrap = True` and breaks. Stop strings (4429) and stop tokens (4398) also break.
- The `finally` at 4472 calls `gen.close()`. That runs `Engine.generate`'s `finally` (2020-2047): it sends `STOP`, then
  drains to `DONE` and **discards every `T` line** that arrives after the break. `Engine.generate` also returns at 1987-1990
  on a `T` line once `cancel` is set.
- The continuation is `prompt = prompt + seg + extra` (4577). `seg` is exactly the tokens consumed before the break, and
  `extra` is `REASONING_WRAP_UP` tokenized (~17 tokens). The overshoot tokens are never in it.

**Reuse decision** (1355-1379): reuse needs `lcp == hist.size()` (1358). Otherwise it falls back to the snapshot (1377,
`hist.resize(snap_len)`) and re-feeds from `snap_len`.

**The measured logs match this exactly.** Source: `~/.cache/xolotl/runs/dspark/serve-dspark{,2,3}.console`, identical in
all three runs.

| request | prompt | engine emitted | front end consumed (= lcp - prompt) | hist before decision | lcp | result |
|---|---|---|---|---|---|---|
| 1 prose | 69 | 355 | 350 | 69 + 354 = 423 | 419 | snapshot: reused=69, 367 re-fed as a batch (10-12.6 s) |
| 2 code | 72 | 353 | 350 | 72 + 352 = 424 | 422 | snapshot: reused=72, 367 re-fed (11-13 s) |
| 3 code t0.7 | 66 | 351 | 350 | 66 + 350 = 416 | 416 | **reused=416**: the cut fell on a burst's last token, 17 fed as passes (2.3 s) |

Why the log said `hist=69`: the `serve: reuse ...` line (1380) is printed **after** the branch. The snapshot branch had
already done `hist.resize(snap_len)` (1378). The pre-decision `hist` was 423, so `hist=69` is just `snap_len`.

Why `snap=69` rather than `snap=66` (plain logs `snap=66` for its 72-token prompt): with DSpark the request's tail runs
as the DSpark seed, and `STRATA_SERVE_SNAP_BACK` is switched off then (1549:
`if (snap_back > 0 && sp_tail == 0 && bcap == 0 && tail_keep == 0)`). So in DSpark mode the snapshot always sits at the
prompt's end. See §6.3.

**Plain, for comparison** (1852-1864): the STOP check comes before each sample. A token is pushed after its `step()`.
Printing token N is followed by `step(N)` (~120 ms) before the next STOP check, so the front end's STOP normally arrives
in time and `hist` = exactly the consumed tokens (`350 tokens (cancel)`, `reused=422`). Plain overshoots only if the
front end takes longer than one token time to send STOP. Then it overshoots by 1 and falls back to the snapshot the same
way (§6.4). The patch does not change plain.

## 2. State inventory (what a rewind to position s has to respect)

From `ds41_dense.cpp` `struct Layer` (155-170), `upload_inputs` (1293-1400), `snapshot` / `park` (1794-1858),
`ds41_dspark.cpp`, and `ds41_engram.cpp`.

| state | indexing | after a rewind to s (positions >= s re-fed) |
|---|---|---|
| compressed KV `comp` (ratio 1: row p; ratio 2: row p/2, open groups write the spare row `cmax`) | positional | OK. A query at q sees rows < (q+1)/r (1339), all of them re-fed by then. The spare row is never ranked (`pass_cap` <= cmax, 1275). |
| indexer keys `idxk`, candidate blocks (`i_lastblk`, per query) | positional / per pass | OK, as above. There is no persistent block summary. |
| `sel` / `selm`, `x_state` / `pre_state`, `i_*` | per-pass columns | OK, rebuilt every pass. |
| Engram token history `hist_` | positional (`set_token(pos)`, engram.cpp:248) | OK. The hash at s reads s-3..s-1 (< lcp, unchanged). |
| **window K/V ring `raw`** [DH, RAW = SWA + 16 = 144] | ring, slot = pos % 144 (dense.cpp:1138-1141, 1320) | A query at q >= s reads positions [q-127, q-1] < s from the ring. A write at position w evicts w - 144, which is needed iff w - 144 >= s - 127, i.e. **w >= s + 17**. **Rule: every position written since then is <= s + 16** (`RAW - SWA`, the same slack the DSpark verify relies on). |
| **ratio-2 compressor state `st_kv` / `st_sc`** [DH, 2] | ring by parity; a pass writes the states of its last 2 tokens (1353-1365) | For an even s, the group (s, s+1) is inside the pass, so the ring is not read. For an odd s, token s needs s-1's state from slot 0, which holds it only if no pass wrote a position >= s afterwards. **Rule: odd s and something written >= s means re-feed s-1** (= the DSpark loop's `refeed = dirty && ((P + 1) & 1)`, 1782). |
| snapshot `raw_snap` / `st_*_snap` + `snap_len` | copy of both rings at snap_pos | Untouched as long as nothing below `snap_len` is rewritten. **Rule: s >= snap_len.** |
| park (`park_hist`, parked rows) | host copies | Not involved. The existing `park_rows` check (1350) still applies. |
| DSpark drafter window (`filled[slot] = pos`, slot = pos % 128, visible iff in (p-128, p]; dspark.cpp:46, 417, 436) | positional | Exact after dropping entries >= s (new `truncate(s)`). Drafts are verified, so the drafter state changes acceptance only, never which tokens are possible. |
| `dsp_next` (the drafter's next position) | scalar | Resume < `dsp_next`: truncate instead of reset. |

So rewinding the *live* caches (no `restore()`) to s is exact (the same tokens at the same positions, rows computed
from them) iff:

1. s <= lcp, s <= prompt - 1 (at least one token to feed), and s >= max(snap_len, 1);
2. s is even, or no position >= s has been written (otherwise use s - 1);
3. the highest position written since the rows [s-127, s-1] were written is <= s + 16.

Rows in [s, lcp) are re-fed with the same tokens. They can differ from the originals only by pass-shape numerics, the
same as any DSpark verify vs. prefill.

## 3. Options

- **(b) emit the drafts one by one and check `stop_pending()` between them: rejected, it does not work.** The front end
  sends STOP only after it has read and processed token N (detokenize, parse, budget). The engine prints a whole burst
  within microseconds, so drafts N+1.. are already in the pipe before any STOP exists. Checking after each pass would
  catch only a cut on the burst's head token. Request 2 above (2 extra) and the common mid-burst cuts stay broken.
  Making it exact would take a per-token handshake, i.e. latency on every token.
- **(c) the front end includes the overshoot: rejected.** It covers only the budget path. For a stop string the tokens
  after it must not be in the next prompt. A disconnect's next prompt is the client's, not ours. It also changes visible
  output (thinking runs 1-14 tokens past the budget, `completion_tokens` changes) and needs the drain to keep the `T`
  lines plus re-sync of the parser and detokenizer.
- **(a) the engine resumes at lcp, with the re-feed rule from §2: chosen.** It is the same situation as a rejected draft,
  which the engine already handles (rows past the restart are stale, the window has 16 slots of slack, an odd restart
  re-feeds one token). It covers every cause (budget, stop strings, disconnects, lookup) without touching the front end.
  It is DSpark-only by construction. Instead of a fixed "hist.size() - lcp <= kVerify + 1", it tracks the **highest
  position actually written** (`dsp_whi`), so the window bound is exact and conservative.

## 4. The patch (`docs/CANCEL_REUSE.patch`, `patch -p1` from the repo root, 5 files)

- `ds41_dense.{hpp,cpp}`: `int window_slack() const` returns `RAW - SWA` (16), documented with the rule.
- `ds41_dspark.{hpp,cpp}`: `void truncate(int pos)` sets `filled[slot] = -1` for every slot holding a position >= pos.
- `ds41_generate.cpp` (serve loop only):
  - `dsp_whi` is the highest position written to the live caches. It starts at `total - 1` after the prompt (or the
    carried value when the request itself was a live-cache resume). It is raised after every `pass_v` (`p0 + vt.size() - 1`)
    and set to `pos - 1` after a plain decode in DSpark mode. It is -1 after any error and always -1 without
    `STRATA_SERVE_DSPARK`.
  - Reuse decision: when `dsp_whi >= 0` (DSpark mode, no unpark, no partial unpark), compute s by the §2 rule and resume
    there: `hist.resize(s)`, `reused = s`. This **replaces** the `lcp == hist.size()` branch for DSpark histories. That
    branch also lacked the odd-position re-feed (§6.1). If the rule says no, the old snapshot / park / new-conversation
    branches run unchanged.
  - Drafter: if the resume is below `dsp_next`, call `truncate(s)` instead of `reset()`. The window keeps the thinking it
    had, and the tail then appends from s.
  - New log line when the resume is not a plain extension:
    `serve: DSpark resume at S (lcp L of hist H, passes wrote up to W)`. The existing `serve: reuse ...` line is
    unchanged. As before it prints `hist` after the decision, so it now shows `hist=S`.
  - Kill switch `STRATA_SERVE_DSPARK_RESUME=0` gives the exact old rules (for A/B).
- **Plain is untouched.** Every new branch needs `dsp_path` (= `STRATA_SERVE_DSPARK`) or `dsp_whi >= 0`, which is only
  ever set when `dsp_path` is set. With `dsp_path == nullptr` the chain is the old one: `!dsp_hist` is always true. The
  plain decode loop itself is not edited.

Expected on the measured requests: request 1, lcp 419 is odd and positions >= 419 were written, so it resumes at **418**
(18 tokens fed). Request 2 resumes at **422** (17). Request 3 is unchanged at **416**.

## 5. Verified in code vs. still needs a run

Verified by reading:

- The overshoot mechanism, and the front end dropping drained tokens.
- `hist=69` being post-resize, and `snap` = prompt end in DSpark mode.
- The window-ring math: `RAW = SWA + 16`, the slot formula, and the w >= s + 17 eviction bound.
- The ratio-2 ring layout and its write rule, and that the spare row is never read.
- Engram, compressed, indexer and drafter state being positional.
- That the patch applies to the current tree.

Needs a build and a run:

- That it compiles.
- That `reused` matches the numbers above.
- The numerical check (test 3).
- That plain is bit-identical (`verify_setup`).
- The TTFT gain.

## 6. Other findings on the way (not fixed by the patch unless stated)

1. **Latent bug (fixed by the patch, DSpark only):** the old `lcp == hist.size()` reuse after a DSpark decode resumed at
   an odd s without re-feeding s-1. If the last pass had rejected drafts (wrote >= s), slot 0 of the ratio-2 state ring
   held a rejected draft's state instead of s-1's. Group (s-1, s)'s compressed row (ratio-2 layers) was then built from a
   wrong partner state. It was silent: no error, slightly wrong KV.
2. **Latent bug, plain AND DSpark, NOT in the patch: partial unpark** (1328-1338, `STRATA_SERVE_PARK_PARTIAL`, default
   on). `unpark()` (dense.cpp:1843) writes back the *snapshot copies* (`raw_snap`, `st_*_snap`) and the positional rows,
   but the partial branch never calls `restore()`. The resume at `lp` therefore reads the **side request's** window ring
   and ratio-2 ring for the first ~128 positions. Even with `restore()`, a gap `park_size - lp` above 17 (lp_min allows
   25% of the park) breaks the window rule. Suggested fix, as a separate change because it changes plain:
   - in the partial branch, first set `lp -= (lp & 1) && lp + 1 < park_hist.size()` (after a restore the ratio-2 ring
     holds S-2 and S-1);
   - then require `park_hist.size() - lp <= dense.window_slack() + 1` (the ring is the snapshot's, written up to S - 1);
   - call `dense.restore()` after `unpark()`;
   - otherwise fall back to the full re-read.

   The measured case (run19, lp = S - 1) fits that rule. It is unverified whether outputs were visibly affected: compare
   `STRATA_SERVE_DUMP` logits of a partial-unpark turn against a fresh read.
3. **DSpark mode loses `STRATA_SERVE_SNAP_BACK`** (1549). The snapshot sits at the prompt's end, so a client that drops or
   re-renders its answer (the 2026-10-24 157K-token re-reads) misses it and re-reads everything (or parks). The patch
   does not cover this, because lcp < snap_len there. Possible fix: split the DSpark tail at `snap_at` and take
   `dense.snapshot(snap_at)` between pass_v chunks (the batch-tail variant would end the batch at `snap_at`). Worth doing
   before DSpark becomes the default.
4. **Plain overshoot race:** a front end slower than one token time gives `hist` = consumed + 1, so a full snapshot
   fallback. The same rule would fix it with no re-feed: plain writes exactly `hist`, so `w = s` and the ring holds s-1.
   Left out because plain must not change.
5. **A draft that is a stop token is pushed into `hist`** (1837 pushes before `emit`). Plain never pushes its stop token.
   It is harmless for reuse (the next prompt normally contains the EOS too), but it is an asymmetry. A draft hitting
   `max_new` also leaves `hist.size() < P + 1`. The resume rule handles both (`dsp_whi >= P`).
6. **`STRATA_SERVE_LOOKUP`:** same mechanism, a larger overshoot (up to 14 drafts a pass, ~96% accepted on copies), so up
   to ~14 tokens in `hist` past the cut. Usually within the 16-position slack. A worse case falls back to the snapshot,
   which is correct.
7. **Client disconnect** (4467 / 4589) and **stop strings** (4429): the same STOP + drain path, so the same overshoot. A
   retry that resends the original prompt hits the snapshot in both modes. A continuation that keeps the partial answer
   now resumes in DSpark mode. In plain it is unchanged (snapshot unless exact).

## 7. Test plan (server stopped, MemAvailable >= 110 GiB)

0. `dv -s "pre cancel reuse"`, then back up `build-strix/ds41_generate` to `ds41_generate.pre-cancelreuse`. From the repo
   root: `patch -p1 --dry-run < docs/CANCEL_REUSE.patch`, then `patch -p1 < docs/CANCEL_REUSE.patch`, then
   `scripts/build_xolotl`. `ds41_dense.hpp` changes, so more units recompile than usual.
1. **Plain bit-identity:** `scripts/verify_setup` gives PASS.
2. **Reuse:** run `mkdir -p ~/.cache/xolotl/runs/cancel/dumpA`, then
   `STRATA_SERVE_DUMP=~/.cache/xolotl/runs/cancel/dumpA scripts/start_xolotl_dsv41 --model 1017 --dspark`. Then run
   `scripts/dev/serve_ab.py cancelA`. The dump directory must exist (ofstream does not create it). In the engine log
   (console), check:
   - req 1: `DSpark resume at 418 (lcp 419 of hist 423, ...)`, `reuse lcp=419 ... hist=418 snap=69 prompt=436 -> reused=418`,
     `prompt phases: 18 new = 0 batch ... DSpark tail 18 as passes`;
   - req 2: `reused=422` (17 new);
   - req 3: `reused=416`;
   - the continuations' TTFT ~2-3 s instead of 10-13 s, and no `decode failed` or `ERR`.

   Request 1's first pass (355 tokens) runs before anything changed and must match the old console text exactly.
   Request 2 is greedy and a fresh conversation, so its first pass is expected identical too. Request 3 (t = 0.7) can
   differ: `spec_accept` draws from the rng once per checked draft, so different pass structure in requests 1-2 shifts
   its rng stream. Its lcp may then not be 416, and the run is not a failure for that alone.
3. **Numerics A/B on the same binary:** restart with `STRATA_SERVE_DSPARK_RESUME=0 STRATA_SERVE_DUMP=.../dumpB` (the old
   rules) and run `serve_ab.py cancelB`.
   - `cmp dumpA/req000.f32 dumpB/req000.f32` must be bit-identical. req002 should be too (see the caveat in step 2).
   - req001 / req003 are the continuations' last-row logits, A = resume, B = snapshot + batch re-prefill of 367 tokens.
     Compare with numpy (`np.fromfile(..., np.float32)`): same argmax, and max|Δlogit| / KL(softmax) at the size of a
     batch-vs-pass difference. For that noise floor, rerun B with `STRATA_SERVE_BATCH_MIN=0` (the 367 as passes) and
     diff B vs B'.
   - Then compare the 333-token continuations: they should agree to roughly the ~1-3% near-tie level that DSpark already
     has. Large disagreement right after the cut would point at a ring error.
4. **Optional, odd-extension path (§6.1):** this needs a direct engine run. Use a copy of
   `xolotl/tools/v41/serve/drive_engine.py` with the strixhalo paths (`start_xolotl_dsv41 --dry-run` prints the engine
   command), and change its `extend` so it keeps only the first `stop_after` tokens. Sweep `stop_after` over a few odd
   and even values and check every continuation's `req*.f32` against a fresh read of the same ids (an unrelated request
   in between forces `reused=0`).
5. Kill switch available: `STRATA_SERVE_DSPARK_RESUME=0`.
