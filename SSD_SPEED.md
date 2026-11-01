# SSD speed and decode speed on Strix Halo

As of 2026-10-28 (Samsung 990 Pro, plain and DSpark measured). 

DeepSeek-V4.1-Flash decode on a 128 GB Strix Halo is drive-bound: the experts (243 GB) do not fit in RAM, so every
token reads some of them from the NVMe. Measured on a **Samsung 990 Pro 1 TB** (PCIe 4.0 x4): **~9 tok/s plain** and
**~11 tok/s with DSpark** (`--dspark`), with the same model and output distribution. A slower drive costs speed
directly; two fast drives in RAID 0 are estimated at ~9.5 tok/s plain.

## Where the time goes

About 6,000 experts (80-86 GiB) stay in RAM; the rest are read from the SSD when the router picks them. Each such
"miss" reads one 14.5 MiB expert.

| Part of one decoded token (plain decode, 990 Pro) | Time |
| --- | --- |
| Dense half (attention, router, shared expert) | ~58 ms |
| Experts already in RAM | ~20 ms |
| Expert reads from the SSD: ~12.8 misses x ~2.5 ms | ~32 ms |
| Other | ~2 ms |
| **Total** | **~112 ms = ~9 tok/s** |

Measured on GSQ-217 in quality mode with the read-path defaults, on a 15K-token agent prompt plus 3,000 decoded
tokens (warm window, two runs: 8.95 and 8.93 tok/s; misses identical run to run). With zero read cost a token would
take ~83 ms, about 12 tok/s plain: that is the ceiling a drive can buy. DSpark (~11 tok/s) checks ~3 drafted tokens
per pass, so it reads the experts of several positions at once and depends on the drive the same way.

## Estimated decode speed by drive

Plain decode by drive. Rows marked measured are measured; the others apply the same token budget with another read speed.

| Drive's sustained throughput on 14.5 MiB random reads | Time per miss | Decode speed |
| --- | --- | --- |
| **Samsung 990 Pro 1 TB** (PCIe 4.0 x4, heatsink) | **2.5 ms (measured)** | **8.9 tok/s (measured)** |
| Kingston OM8TAP42048K1 2 TB (DRAM-less, PCIe 4.0 x4): 2.9 GB/s, dropping to 0.5-0.8 GB/s in ~18 s slow phases | 7.4-8.4 ms (measured, older read path) | ~5 tok/s (measured, older read path) |
| 2.9 GB/s without slow phases | ~5.7 ms | ~6.4 tok/s (est.) |
| 4 GB/s | ~4.3 ms | ~7.2 tok/s (est.) |
| 6 GB/s | ~3.0 ms | ~8.2 tok/s (est.) |
| 10 GB/s (PCIe 5.0 class) | ~2.0 ms | ~9.2 tok/s (est.) |

The estimate: 83 ms of fixed work per token, plus 12.8 misses, each costing the read time of 15.2 MB plus ~0.5 ms overhead.

```
tok/s ≈ 1000 / (83 + 12.8 × (15.2 / GB/s + 0.5))
```

## Two drives in RAID 0 (example: 2 x Samsung 980 Pro)

Two PCIe 4.0 drives in RAID 0 should roughly double the read bandwidth, to an estimated 10-12 GB/s on these reads, not the 14 GB/s of two spec sheets. Each expert is read as three 4.83 MiB pieces in 32 parallel parts, so with a 512 KiB stripe every read keeps both drives busy.

| Setup | Throughput on these reads (est.) | Time per miss | Decode speed (est.) |
| --- | --- | --- | --- |
| 1 x 980 Pro | ~5.5-6.5 GB/s | ~2.8-3.3 ms | ~8.0-8.4 tok/s |
| 2 x 980 Pro, RAID 0 | ~10-12 GB/s | ~1.8-2.0 ms | ~9.2-9.6 tok/s |
| 2 x 980 Pro at the full 14 GB/s spec | 14 GB/s | ~1.6 ms | ~10 tok/s |

The 14 GB/s is not reached because 7,000 MB/s is the sequential peak, while good PCIe 4.0 drives typically do ~5.5-6.5 GB/s on large random reads (review figures, not measured here). Past ~10 GB/s the ~83 ms of fixed work per token dominates, so a third drive would add little.

What the setup needs:

- **Each drive on its own PCIe 4.0 x4 link.** RAID 0 runs at the pace of its slowest member, so keep slower links out of the array. On our Minisforum MS-S1 MAX that means the primary M.2 slot plus the x16 slot (wired x4) with a single-M.2 adapter; the x1 M.2 slot holds the OS drive.
- **Check the links** after installing: `/sys/class/nvme/nvme*/device/current_link_speed` should read `16.0 GT/s` and `current_link_width` `4`.
- **mdadm RAID 0 with XFS or ext4** for the model folder: direct reads are predictable there, with no compression or checksum work.
- **Heatsinks and current firmware.** The 980 Pro is known to throttle without a heatsink, and decode reads for minutes at a time.

Without RAID, the engine can also split each expert read across two copies of the model file, one per drive. That path exists but is untested on this box.

## What to look for, and how to test your drive

Spec-sheet speeds are sequential reads and do not predict this workload. What counts is sustained reads of ~5 MB pieces scattered across a 243 GB file, for minutes at a time.

- **Sustained random reads of large blocks.** Look for reviews that test random reads of 1-4 MB blocks, or sustained reads over a full drive, not a short burst.
- **No slow phases.** Our first drive (Kingston OM8TAP42048K1, PCIe 4.0 x4, DRAM-less) dropped to 0.5-0.8 GB/s for ~18 s at a time, for 14-34% of decode time. Its temperature reading stayed at 55-57 C, so the cause is open. The 990 Pro (with its own DRAM cache) showed no slow phases at 46-47 C under decode.
- **Cooling.** A heatsink or airflow on the M.2 slot rules out thermal throttling as a cause.
- **A second M.2 slot**, if the board has one, for two drives sharing the reads.

To test a drive, read the model file itself for 60 s:

```
fio --name=experts --filename=V41-1017-GSQ-217.gguf --readonly --direct=1 \
    --ioengine=io_uring --rw=randread --bs=4M --iodepth=16 --runtime=60 --time_based
```

On the Kingston fio reported 3.45 GB/s while the engine's 14.5 MiB expert reads got ~2.9 GB/s. Take ~85% of fio's figure into the table above. On btrfs the model folder must have compression off (`chattr +m`), or direct reads are not measured correctly.

## How sure this is

That a faster drive speeds up decode is measured; how much is estimated.

Measured:

- **The drive is the limit, not the software.** Reading 2, 4 or 8 experts at once costs the same 5.1-5.3 ms each. Thread count, queue depth, io_uring and chunk size change nothing. The engine adds ~10-15% on top of the raw read.
- **Slower reads make slower tokens.** Identical runs with the same 18.0 misses per token took 128 ms or 164 ms of expert time, depending on how long the drive spent in slow phases.
- **Output is unchanged.** Routing and tokens are bit-identical across runs; a drive changes only speed.

Estimated:

- The 990 Pro is measured and matches the token budget; RAID 0 and other drives are not tested. The table assumes ~0.5 ms overhead per miss and the same miss count.
- The miss count assumes 128 GB of RAM; less RAM means more misses.
- Long new prompts also read experts from the SSD, so prefill should speed up too, by an amount not measured.

The 990 Pro row replaces the estimate for its class; the RAID 0 rows are still estimates.
