# MTP / DSpark expert pre-activation (NVMe read-ahead) - offline evaluation (2026-10-28)

Question: can the DSpark draft stack (or anything else known before a verify pass needs it) predict which routed
experts the next verify pass will miss in the RAM arena, so their 14.5 MiB NVMe reads start early and the file wait
is hidden? Target: Strix Halo, GSQ-217 on the 990 Pro (~6 GB/s, ~2.5 ms per blob), DSpark best config (depth 4,
`STRATA_DSPARK_CONF=0.6`, `STRATA_DSPARK_EXP_TYPE=iq3_xxs`, arena 80 GiB + ~9 GiB extension).

**Verdict: not worth building now.** A perfect predictor would hide ~25-33% of a verify pass. The best realistic
predictor (the trunk's own router applied one layer early to every pass row) covers only 30-45% of the misses, and
its 70-80% wrong guesses use the same drive the real misses wait on. The timeline model gives **-5 to -9% per pass
(about +5-10% tok/s, ~9.3 -> ~10 tok/s)**, and only if the prediction is fused into an existing graph (≤ 0.05 ms per
layer). With a stand-alone predict graph per layer (measured ~0.35 ms) it comes to about zero. The existing hook
(`STRATA_D3_SPEC`) measured **no gain at n=8, +11% slower verify passes at n=16 / 2 ahead, and it is not lossless**:
the token stream changed at token 69. The drafter's
own signals (token ids, embeddings, late hidden states) predict the trunk's experts worse than the router one layer
early does (§3). Plain decode: about -4-5% at best (same model). Main risk: wasted reads on a drive that is already
the bottleneck, plus the drive's slow phases.

Measured vs estimated is marked in every section. Scripts: `scripts/dev/prefetch/` (all offline except the engine
runs below). Data: `~/.cache/xolotl/runs/prefetch/` (local disk).

## 1. Data

| Run | What | Source |
|---|---|---|
| T1 (2026-10-25, existing) | plain decode, warm bench prompt (14.8K) + 3000 forced tokens, `STRATA_DEC_TRACE` + `STRATA_FILE_PRED=1` (router-ahead top 16, k = 1..4, for every layer and token) | `runs/nvme/T1.trace` |
| R1 `pf-r1` | DSpark best config, warm prompt, 400 tokens, `STRATA_ROUTE_TRACE` + `STRATA_DSPARK_CONFLOG` + `DS41_PROF_X=2` | `runs/prefetch/r1.*`, `runs/dspark/pf-r1.log` |
| R2 `pf-r2-spec8a1` | R1 + `STRATA_D3_SPEC=8 STRATA_D3_SPEC_AHEAD=1` (existing read-ahead hook) | `r2.*` |
| R3 `pf-r3-spec16a2` | R1 + `STRATA_D3_SPEC=16 STRATA_D3_SPEC_AHEAD=2` | `r3.*` |
| R4 `pf-r4-plain` | plain decode, warm prompt, 400 tokens, `STRATA_DEC_TRACE` + `STRATA_DUMP_MAIN` + `STRATA_FILE_PRED=1` (late-state proxy, §3.4) | `r4.*` |

All four engine runs used `flock ~/.cache/xolotl/gpu.lock`. DSpark is deterministic at a fixed config: R1's 400 tokens
are identical to the first 400 of the 2026-10-28 `iq3-c06` run.

## 2. Where the file wait sits in a verify pass (measured, R1)

- 137 passes, 3.37 rows/pass (incl. re-fed rows), 2.91 emitted tokens/pass. **49.3 file misses per pass = 16.9 per
  emitted token**, ~1.2 per layer call. File wait 115 ms/pass (2.47 ms/blob). After the first 20 passes (graph
  warm-up): 43.5 misses/pass, verify = 113 ms + **3.34 ms per miss** (least squares over passes). The marginal cost of a miss
  is above the 2.5 ms read: it also pays its FILEG kernel and the sum after the reads.
- Per layer call: 39% have no miss, 29% one, 15% two, 8% three, 4.6% four, 3.6% five or more (max 9). **51% of the misses
  sit in calls with ≥ 3 misses**: bursts that one layer of lead (≈ 7.5 ms ≈ 3 blobs) cannot fully cover.
- Misses by the first pass row that routes them: row 0 29%, 1 27%, 2 20%, 3 14%, 4 10%. **11-12% of the misses come only from
  rejected draft rows**: experts read for rows that are then thrown away. A rejected row's experts are admitted to the
  arena, so when the next pass routes them they are already hits. DSpark already acts as a small prefetcher.
- Per-call profile (`DS41_PROF_X=2`, the 2026-10-28 c06b): launch 3.74 ms of which ~3.5 is waiting for the file reads, sum+sync
  1.54 ms, classify/xq 0.05 ms. With no file wait a layer's tier would be ~1.8 ms + whatever hit-kernel GPU time the
  reads now hide (unknown, ≤ ~1 ms).
- The drive is busy ~38% of a pass (48 × 2.5 ms of ~310 ms), but only while a layer waits on it. The ~62% idle time is the
  budget for read-ahead.

## 3. Candidate predictors: recall / precision (measured offline from traces)

Scoring (`pred.py`): a miss = a routed expert outside the arena, with the arena replayed by `scripts/dev/nvme/sim.py`
(`lfur:0.995` + 635 extension slots = the served `STRATA_ARENA_LFU` / `STRATA_ARENA_EXT_GIB`; 12.64 misses/token on
T1's window ≥ step 1024, matching the live ~12.8). A read is issued for every predicted expert outside the arena.
Recall = misses predicted / misses, precision = useful reads / reads issued.

### 3.1 Per token (plain decode, T1, 1975 tokens)

| predictor | when known | recall | reads/token | precision | recall L0-9 / 10-19 / 20-29 / 30-39 |
|---|---|---|---|---|---|
| r1.2: router of L on the state entering L-1, top 2 | 1 layer early | 17.5% | 5.3 | 42% | 13 / 17 / 24 / 18% |
| r1.4 | 1 layer early | 32.8% | 16.6 | 25% | 23 / 34 / 44 / 35% |
| r1.8 | 1 layer early | 51.8% | 56.4 | 12% | 34 / 55 / 68 / 55% |
| r1.16 | 1 layer early | 66.1% | 175 | 4.8% | 47 / 71 / 82 / 70% |
| r2.4 / r2.8 | 2 layers early | 28.4 / 43.2% | 20 / 63 | 18 / 8.7% | |
| r3.4 / r4.4 | 3 / 4 layers early | 26.0 / 21.8% | 23 / 26 | 15 / 11% | |
| tok.6: experts this token id routed the last times it was decoded | pass start | 13.9% | 6.0 | 29% | 17 / 14 / 13 / 12% |
| tok.12 / tok.24 | pass start | 20.2 / 24.6% | 16 / 37 | 16 / 8.5% | |
| emb.4: token embedding through layer L's router | pass start | 16.1% | 77 | 2.6% | 14 / 17 / 23 / 13% |
| emb.8 / emb.16 | pass start | 20.1 / 25.7% | 173 / 370 | 1.5 / 0.9% | |
| r1.4 + tok.6 | | 40.2% | 21.7 | 23% | 33 / 42 / 47 / 41% |
| r1.8 + tok.6 | | 57.0% | 61 | 12% | |

Per-layer values: `runs/prefetch/t1_pred.json` (`recall_by_layer`, `reads_by_layer`).

### 3.2 Per verify pass (pseudo passes)

T1 grouped into passes of acc+1 accepted rows, using R1's acceptance sequence (2.91 rows/pass), one arena call per
layer and pass on the union (as `gpu_run_n_ex`). The rejected rows are missing (a plain trace has none): 36.6
misses/pass here vs R1's 43.5. That is the ~12% rejected-only share in §2 plus the different text.

| predictor (union over the pass rows) | recall | reads/pass | precision |
|---|---|---|---|
| r1.2 | 18.2% | 14.7 | 45% |
| r1.4 | 34.1% | 45.3 | 28% |
| r1.6 | 46.0% | 91 | 18% |
| r1.8 | 53.8% | 148 | 13% |
| r2.4 | 29.9% | 54 | 20% |
| tok.6 | 13.8% | 15.8 | 32% |
| emb.8 | 23.5% | 444 | 1.9% |
| r1.4 + tok.6 | 41.3% | 58 | 26% |

Taking the union over rows changes neither recall nor precision much compared with single tokens. Each row adds its
own wrong guesses, and the pass has proportionally more misses.

### 3.3 What each MTP-side idea gives

- **Token-only guesses (the draft ids are known when the drafter finishes, a whole pass ahead).** DSv4.1-Flash has no
  hash-routed (token-id) layers: V4-Flash had three, and V4.1's 40 × 384 layers all route with the learned gate
  (`ds41_geometry.hpp`, no `tid2eid`). Engram is token-keyed, but it is a memory table, not expert choice. The
  embedding through the routers (`emb`) is useless: precision 1-3%, because a layer's routing depends on context
  through the attention and the hyper-connection mix. A per-token-id history (`tok`) is the best token-only signal
  (29% precision at 6 reads/token). It was measured with decode-only history; history from the 14.8K prompt's prefill
  routing would help (not measurable with the existing traces, which start after the prompt).
- **The previous pass's routing of the same positions.** Only a rejected position is recomputed, and with a different
  token. Its experts were admitted when the previous pass routed them, so whatever overlaps is already a hit and needs
  no read (§2). An accepted, re-fed row repeats its exact routing, so the same applies.
- **Drafter hidden states through the trunk's layer-L router.** The drafter reads and writes late-layer states (the HC
  mean entering layers 37-39). R4 bounds this optimistically: the trunk's own true late state of the same token through
  router L. Result in §3.4. Even the router applied 2-4 layers early (same residual stream, much closer to layer L) falls
  to 22-28% recall at top 4.
- **Router one layer early on the trunk state** (`r1`, = `STRATA_D3_SPEC_AHEAD=1` / `STRATA_FILE_PRED k=1`): the best
  signal, and the only one with useful precision at a small budget.

### 3.4 Late-state proxy for the drafter hidden (R4, measured)

`main_proxy.py` on R4 (plain decode, 400 tokens + the 127-token prompt tail, window from step 160, 15.2 misses/token):
the HC mean of the trunk state leaving layer 36 or 38 (what the drafter reads; its own output is an estimate of
it for the next position), rms-normed, through layer L's router, top N.

| state used | N | miss recall | reads/token | precision | recall L0-9 / 10-19 / 20-29 / 30-39 |
|---|---|---|---|---|---|
| h36 of the SAME token (oracle: the drafter can only approximate it) | 4 | 17.1% | 44 | 5.8% | 2.5 / 9.2 / 28 / 33% |
| h36, same token | 16 | 36.1% | 235 | 2.3% | 7.6 / 22 / 53 / 70% |
| h38, same token | 4 | 15.0% | 52 | 4.3% | 2.4 / 7.3 / 23 / 31% |
| h36 of the PREVIOUS token (known at the pass start) | 4 | 1.5% | 42 | 0.5% | ~chance |
| h38, previous token | 16 | 6.7% | 254 | 0.4% | ~chance |
| for comparison, same run: r1.4 (router 1 layer early) | 4 | 36.9% | 18 | 30.4% | 26 / 37 / 49 / 39% |

Even the true late state of the same token predicts the early and middle layers' routing hardly better than chance,
and only reaches 5% precision at the top. A drafter hidden state (an approximation of that state, for the next
position) through the trunk routers is dead as a predictor. R4 also confirms the router-ahead numbers on the 2026-10-28
config (r1.4: 37% recall / 30% precision vs T1's 33 / 25%). `tok` is weak in R4 (1.9% at 6) because 400 steps give
little history: the T1 values need ~1000+ decoded tokens of history.

## 4. Timeline model: how much file wait can be hidden (estimated)

`timeline.py`, on the T1 pseudo passes with the real predicted ids. Per pass: dense (attention + router) 110 ms and
tier without file wait 81 ms (`iq3-c06`: 110.4 / 200.2 - 119), spread over the 40 layers. Logits 6 ms, drafter 15 ms. One
device serves blobs back to back at 2.5 ms, demand reads first. A read-ahead blob is read in 4 parts, so a demand read
waits at most for one part (the engine splits blobs over its reader pool); 1 part = no preemption. Read-ahead for
layer L is issued at layer L-K's start, and cancelled if it has not started when L runs. At most 64 blobs in flight.
Separate stash (`--admit 0`); §5 covers landing in arena victim slots. Predict cost: 0.05 ms/layer (fused into the
previous layer's finish graph) or 0.25 ms (a stand-alone `predict_multi`). The model's baseline is 303 ms/pass with
91 ms file wait (36.6 misses). R1 measured ~115 ms with the rejected rows, so scale hidden ms by ~1.25.

| config (4 parts, fused predict 0.05 ms) | ms/pass | vs none | file wait ms/pass | misses covered | read-ahead/pass (wasted) | drive busy |
|---|---|---|---|---|---|---|
| none | 303.4 | | 91.4 | | | 30% |
| oracle 1 layer ahead (exact routing) | 228.3 | **-24.8%** | 16.3 | 97% | 35.6 (0) | 40% |
| oracle at the pass start | 225.8 | **-25.6%** | 13.8 | 88% | 32.3 (0) | 41% |
| r1.2 | 289.7 | -4.5% | 75.8 | 18% | 14.7 (8.0) | 39% |
| r1.4 | 280.9 | **-7.4%** | 67.0 | 34% | 45.3 (27.5) | 57% |
| r1.6 | 280.5 | -7.6% | 66.5 | 46% | 91 (46) | 74% |
| r1.8 | 283.5 | -6.5% | 69.6 | 54% | 148 (59) | 84% |
| u1.2 (pass-ranked union, ≤ 2 per layer) | 281.9 | -7.1% | 68.0 | 30% | 77 (66) | 91% |
| r2.4 | 284.5 | -6.2% | 70.6 | 30% | 54 (36) | 64% |
| r1.4 + r2.4 | 282.9 | -6.7% | 67.1 | 40% | 79 (44) | 71% |
| tok.6 | 292.3 | -3.7% | 80.3 | 13% | 15.3 (10.4) | 40% |
| **r1.4 + tok.6** | 276.5 | **-8.9%** | 62.5 | 41% | 58 (31) | 61% |

Same with a stand-alone predict graph (0.25 ms per layer and K): r1.4 -4.8%, r1.4 + tok.6 -6.2%, r1.8 -3.2%. Without
part-level preemption (whole-blob reads): r1.4 -0.8%, r1.8 +4.4%. Existing-hook-like settings (stash 24, predict 0.35
ms): u1.1 -1.1%, u2.2 -2.7%.

Drive speed (same predictors, fused predict):

| read ms/blob | none ms/pass | oracle | r1.4 | r1.8 | r1.4 + tok.6 |
|---|---|---|---|---|---|
| 1.25 (two drives / Gen5) | 257.7 | -15.7% | -5.1% | -6.1% | -6.1% |
| 2.5 (990 Pro) | 303.4 | -25.6% | -7.4% | -6.5% | -8.9% |
| 4.0 (slow phases) | 358.2 | -30.5% | -8.0% | -6.4% | -9.6% |

Reading the numbers:
- Even the oracle leaves ~14-16 ms/pass. That is the bursts: a layer with 5+ misses needs more drive time than the
  one-layer lead gives, and in the model the reads at the pass start queue up in layer order.
- Real predictors stop at -7 to -9%. Coverage grows with the budget, but so do the wasted reads. Above ~50 read-ahead
  blobs per pass the drive is the bottleneck again (busy 75-90%), and the covered reads themselves wait. Precision, not
  recall, is the limit.
- Wasted reads cost drive time, not just slots. A demand read that lands behind a running wrong read waits for it.
  Reading in parts is needed (no preemption: about zero gain).
- Hidden file wait at the best point: ~29 ms of the model's 91 ms/pass, i.e. ~35 ms of the real ~115 ms/pass, about
  **12 ms per emitted token** of the ~40 ms file wait per token.

## 5. Interaction with the arena (estimated)

- **Landing in arena victim slots** (`--admit 1`: every read-ahead blob takes the LFU victim's slot whether used or not):
  misses rise with the wrong guesses: r1.4 +3.6% misses (37.9 vs 36.6/pass), u1.2 +14%. Net: r1.4 -6.8% (vs -7.4% with a
  stash), u1.2 -6.0% (vs -7.1%), r1.4 + tok.6 -8.5%. LFU keeps the damage small, since a never-used blob has the lowest
  count and is the next victim, but it is never free.
- **Separate stash** (`--admit 0`; only used blobs enter the arena, like demand misses): 64 stash slots of 14.5 MiB
  = 0.9 GiB taken from the arena cost +2.2% misses (`--extra 571`: 37.4 vs 36.6/pass), which is in the model's +0.7% on
  the baseline. r1.4 still -7.6% against that baseline.
- **Better: floating slots.** Take the stash slots from the registered arena itself (same `arena.off[]` table). A used
  stashed blob's slot changes owner with the victim's slot (an index swap, as `arena_pick` already does for
  extension slots), instead of today's 14.5 MiB CPU `memcpy` (`fg_copy` in `gpu_run_n_ex`, ~1 ms per blob on the CPU while
  the card waits for the DMAs). An unused one goes back to the pool and evicts nothing.

## 6. The existing hook, measured (R2, R3)

`STRATA_D3_SPEC=n` (ds41_generate.cpp `pass_v`): before layer l's attention, `predict_multi(l + AHEAD)` over the pass
rows, ranked union (score 16 - rank), top n, the ones outside the arena go to `spec_read` (4-thread, 24-slot `SpecStash`).

| run | spec reads issued / used | file misses covered | misses/pass | verify fit (window) | tokens vs R1 |
|---|---|---|---|---|---|
| R1 (off) | - | - | 43.5 | 113 + 3.34 × misses | - |
| R2 n=8, ahead 1 | 3330 / 887 (27% precision) | 11.7% of 7599 | 50.2 | 127 + 3.16 × misses (at 43.5 misses: +2.4%) | **differ from token 69** |
| R3 n=16, ahead 2 | 9808 / 1083 (11% precision) | 14.3% of 7599 | 50.2 | 277 + 2.05 × misses | = R2 (differ from token 69) |

R2 and R3 emitted the same tokens (same 7599 misses), so they are a clean A/B of the read-ahead width at identical
routing: **R3 vs R2: verify 449.5 vs 403.4 ms/pass (+11%), tier 363 vs 233 ms/pass, file wait 22.6 vs 17.9 s**. The
wider guess (9808 reads, 89% wasted) slows the real misses. The stash `take()` waits sit in the tier outside
`file_ms`. This is the §4 "drive saturates" regime, measured.

- R2 vs R1 (different text after token 69): no measurable gain. The verify fit's intercept rises ~14 ms/pass: the
  extra `predict_multi` graph per layer, ~0.35 ms. The model predicted -1% for this setting (§4).
- **Not lossless as it stands.** With the stash open, `gpu_run_n_ex` caps the card's file misses at `kMaxParts` (8) per
  call (`fg_cap`: `... && !gp.spec ? kMaxEnt : kMaxParts`). The excess goes to the CPU pool, whose numerics differ
  (R2: "6 cpu" experts). R1 had 4 of 5480 layer calls with 9 misses. DSpark's acceptance stays exact against its own
  trunk, but the trunk rows are no longer bitwise those of the default path, and the emitted stream changed at token 69
  (DSpark runs are deterministic at a fixed config: R1 = `iq3-c06`).

## 7. What is different from plain decode (vs the NVMe deep dive's "read-ahead ≤ +3.6%, not built")

- Same predictor quality per token (the trunk router; the drafter adds no better signal, §3). The oracle ceiling is the
  same too: ~25% of a token/pass, i.e. the file wait share.
- **For MTP:** (1) a layer of a verify pass takes ~7.5 ms instead of ~2.8 ms, so a one-layer-early read usually arrives
  in time (oracle 1-ahead ≈ oracle at pass start); (2) one predict graph serves ~3.4 rows, so the predict cost per token
  drops ~3x; (3) the drive is now ~6 GB/s (2.5 ms/blob, was 5.3-7.3 on the Kingston).
- **Against:** wasted reads scale with the rows; the pass's union has more misses per layer call (1.2 vs 0.32), so
  bursts are bigger; rejected drafts already pre-admit part of the next pass's experts.
- Model, plain decode on the 990 (`plain.conf`, dense 58 + tier 20 ms/token): oracle -22 to -26%; r1.4 **-5.0%**
  (fused predict 0.05 ms/layer), r1.2 + tok.6 -4.0%; with a 0.25 ms predict graph r1.4 +1.5%. So plain decode would get
  ~+4-5% from a fused, part-preemptive read-ahead (the deep dive modelled -3.6% on the Kingston with whole-blob reads and
  a separate 5 ms/token predict). DSpark ~+7-10%.

## 8. Recommended design (only if a +5-10% lossless step is wanted later)

1. **Predictor:** the trunk router of layer l+1 on the state entering layer l, for every pass row, top 4 per row (r1.4),
   plus the per-token-id history (top 6) seeded from the prompt's prefill routing. Read only experts outside the
   arena. Budget ~45-60 reads/pass. Expected recall ~40%, precision ~25%.
2. **No extra predict graph:** fuse it into layer l-1's finish (`Ds41Dense::finish_predict` exists for n = 1; add an
   n-column variant of `build_finish_predict`, `ds41_dense.cpp`; `predict_multi` stays the fallback). Target ≤ 0.05 ms
   per layer. Otherwise the gain halves (§4).
3. **Reads:** through the persistent `ReadPool` (`ds41_moe.cpp:195`) with a two-level queue: demand parts always first,
   read-ahead blobs in 4-8 parts, unstarted read-ahead cancelled when its layer runs. Replaces the 4-thread
   `SpecStash` (`ds41_moe.cpp:~251`).
4. **Landing:** 32-64 floating arena slots (§5). In `gpu_run_n_ex` (`ds41_moe.cpp:3427`, FILEG classify loop) a stashed
   hit swaps slot ownership with its `arena_pick` victim instead of `fg_copy`'s memcpy.
5. **Losslessness:** keep `fg_cap = kMaxEnt` with the stash open (every file miss on the card, as without read-ahead).
   Gate: R1-style DSpark run, token stream identical to the default, `nvme/tracecmp.py` on a plain run, `verify_setup` PASS.
6. **Plain decode:** `Ds41MoeTier::spec_read` refuses non-D3 calls under `--arena-adapt`; `gpu_run`'s FILE_FAST path
   (`ds41_moe.cpp:2484`) would need the same take-from-stash. The fused `finish_predict_ahead` already exists for n = 1.
7. **Touch points:** `ds41_generate.cpp` `pass_v` (the `STRATA_D3_SPEC` block) and `step` (decode);
   `ds41_dense.cpp` `build_finish_predict` / `finish_predict`; `ds41_moe.cpp` `Ds41MoeTier::spec_read`, `SpecStash`,
   `ReadPool`, `gpu_run_n_ex`, `gpu_run`, `arena_pick`.

**Estimated gain:** DSpark ~-7 to -9% per pass, i.e. **+7-10% tok/s (9.3 -> ~10.0-10.2)**. Plain decode **+4-5%
(8.3 -> ~8.7)**. Both rest on the model's assumptions: zero-cost fused predict, part-level preemption, and per-layer
timing spread evenly. The tok/s noise between runs (±10%, drive slow phases) is larger than the gain, so A/B it on
misses covered, file wait and verify fit, not tok/s.

**Main risks:** (1) precision ~25%: every wrong read takes drive time from the misses it was meant to hide, and in the
drive's slow phases (14-34% of decode on the Kingston, not yet characterised on the 990) the queue backs up; (2) the
predict cost: a stand-alone graph eats half the gain; (3) correctness: the existing path is not bit-identical (§6).

**Bigger levers for the same effort:** fewer misses, not hidden misses. The 11-12% of verify-pass misses that only
rejected draft rows cause (a per-row confidence trim at the layer where reads start, `route_sim.py --prune`). Arena
capacity (+350 experts ≈ -10% misses). A second drive (`STRATA_BLOB_MIRROR`, halves the read time; the read-ahead
gain then shrinks, §4).

## 9. Reproduce

```
cd scripts/dev/prefetch          # python3 -B everywhere (no __pycache__ in the repo)
R=~/.cache/xolotl/runs
G=<folder of V41-1017-GSQ-217.gguf>
python3 -B pred.py $R/nvme/T1.trace $R/ab/warm/force.ids --emb $R/prefetch/emb.npz --passes $R/prefetch/r1.conf --json $R/prefetch/t1_pred.json
python3 -B emb_router.py $G/V41-1017-GSQ-217.gguf $R/ab/warm/force.ids $R/prefetch/emb.npz
python3 -B rtrace.py $R/prefetch/r1.rtrace --from-pass 20
python3 -B timeline.py $R/nvme/T1.trace $R/ab/warm/force.ids $R/prefetch/r1.conf --stash 64 --parts 4 --pred-ms 0.05 [--admit 1] [--read-ms 1.25]
python3 -B main_proxy.py $G/V41-1017-GSQ-217.gguf $R/prefetch/r4.trace $R/prefetch/r4.main
```
Engine runs (each under `flock ~/.cache/xolotl/gpu.lock`): `runs/prefetch/r1.sh`, `r23.sh`, `r4.sh` (`dspark_ab.sh`, warm
prompt, ARENA=80, 400 tokens).
