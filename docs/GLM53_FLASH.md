# GLM-5.3-Flash on this cluster

Background for `recipes/glm-5.3-flash-nvfp4.yaml`.

## Model and checkpoint choice

320B total / 18B active multimodal MoE: 34 KDA linear layers + 11 sparse
NoPE-MLA layers + an MTP head; native context 262,144 (1M capable). Dtype fit
on 121 GiB nodes at TP=4:

| Checkpoint | Total | Per node | Verdict |
| --- | --- | --- | --- |
| bf16 original | ~640 GB | ~160 GiB | impossible |
| official FP8 | ~300 GiB | ~75 GiB | unified-memory knife-edge - avoid |
| LibertAIDAI NVFP4 | 182 GiB | ~46 GiB | the recipe |

## Provenance

Recipe and mod ported from kingjones30/GLM-5.3-Flash-2x-DGX-Spark (MIT,
pinned @48a2f322f5bd), the first published GB10 deployment: 24.74 tok/s code
/ 30.30 structured / 19.58 prose with MTP-5 (~2.55 accept length), 14.6
baseline. Our port changes tensor_parallel 2 -> 4 and the container tag; all
serving flags are theirs verbatim.

`glm5_next` is not in vLLM main; the dedicated arm64 image
`vllm/vllm-openai:glm53-flash-arm64-cu130` (2026-08-26) is required.

## GB10 gotchas (both silent)

1. **FP4 MoE corruption**: the auto-selected FLASHINFER_CUTLASS NvFp4 MoE
   backend produces degenerate repeated-token output on sm_121 with no
   error. `--moe-backend marlin` is mandatory.
2. **Autotune false hang**: a 60+ minute "No available shared memory
   broadcast block found" loop at first boot is FlashInfer autotuning, not a
   hang (CPU >150% = still working). Caches persist to `~/.cache/vllm`;
   later boots are fast.

## The mod

`mods/fix-glm53-nope-rope-pad` adapts GLM's NoPE-MLA to the fp8_ds_mla KV
layout: builds the attention layer with rope=64, zero-pads q/k_pe, and
compacts the kpool indexer top-k table 2176 -> 2048. Required for
`--kv-cache-dtype fp8_ds_mla`.

## Known limits / tuning levers

- `--language-model-only`: multimodal front-end costs ~15.7 GiB on the API
  node; vision stays off until that headroom is proven at TP=4.
- `max_num_seqs: 1` is the validated single-stream config; raising it is the
  first tuning experiment at TP=4 (watch the MLA + KDA state pools).
- MTP-5 speculative decoding is in the base config (validated ~1.7-2x).

## The MM variant (recipes/glm-5.3-flash-nvfp4-mm.yaml)

Replicates MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark (MIT, @aed98a13ca75):
multimodal on, max_num_seqs 8, fp8_e4m3 KV, MTP-4, Ray executor (launch with
--ray), 23-30 tok/s x1 / 72 aggregate x8 measured. The image
(Dockerfile.glm53-mm, sources in docker/glm53/) folds Mia's two layers into
one build: FlashInfer pinned to 0.6.18 SM90-NoPE-MLA (the stock image's
SM120 sparse path assumes pe_dim=64; this checkpoint is NoPE), NCCL 2.30.7,
cutlass-dsl 4.6.2, her 423-line SM121 source patch, Ray 2.58, and the
model's chat template baked at /opt/glm53/.

Her hard-won operational notes: NCCL must be pinned to the CX7 interfaces
with NCCL_IB_GID_INDEX=3 or ncclCommInitRank busy-waits forever;
gpu_memory_utilization 0.84 (0.90 fails the free-memory check on UMA);
--enforce-eager + --moe-backend marlin is the deployed-stable MoE path (her
native path fallback triggers on cudaErrorNoKernelImageForDevice - loud,
unlike the silent FLASHINFER_CUTLASS garbage the text-only variant guards
against).

Choose by workload: text-only single-stream quality/graphs -> the base
recipe; agents with images and concurrency -> this one.

## The EXL3 variant (recipes/glm-5.3-flash-exl3.yaml) - primary

Replicates MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (@c91754f151ce): EXL3
kernels executing inside vLLM's serving layer via Mia's prebuilt overlay
image (ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3, FROM the
dedicated glm53 image; includes the aarch64 AVX-stub compile patch, NoPE
zero-pad into fp8_ds_mla geometry, and video-placeholder fixes).

Why it is primary - all measured by Mia on GB10:
  - 62.9 tok/s x1 / 146.5 aggregate x4: 2.1-2.5x the NVFP4+MTP variants,
    driven by DFlash2 spec decode (k=7, 0.918 acceptance, 6.43 tok/step).
  - Quality: teacher-logit KLD 0.024555 vs official FP8's 0.024629 (1.00x)
    at 54% of the bytes. Checkpoint: brandonmusic tr3-4bpw (164 GiB,
    mirrored as Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw), uniform-K4 routed
    experts. TR3 K6 (254 GiB) exists if quality headroom is ever wanted.
  - 900k context (native-1M "still does not allocate"), fp8_ds_mla KV.

Constraints (Mia's, verbatim spirit): no --moe-backend marlin, no
TRITON_ATTN for drafts, no bf16 KV, no NVFP4 KV (FlashInfer SM12x NVFP4
kernels are dense MHA, not sparse MLA); prefix caching is block-aligned
only. The DFlash2 drafter (incoai/GLM-5.3-Flash-DFlash2) must be present in
every node's HF cache.

## DFlash2 status (updated 2026-08-28)

Contrary to the earlier watch-item assessment, DFlash2 IS running on GB10 -
in the EXL3 stack above, on vLLM, with draft KV forced to bf16/auto and
non-causal draft attention handled by Mia's overlay. The SGLang path
(PR sgl-project/sglang#36708, fa4 draft backend) remains blocked on sm_121.
A DFlash2 drafter for Qwen3.8-Flash-Next exists on the same lineage; porting
this overlay approach to the qwen lane is a candidate speed experiment.

## Measured results - 4x Spark TP=4 (2026-08-30)

First measured TP=4 numbers for both recipes on our cluster (4x GB10,
CRS812 2x200G breakout fabric, dual-rail RoCEv2). Protocol: llama-benchy
0.4.0 (pp2048/tg128, 3 runs, single stream), lm-eval GSM8K 200q 5-shot,
lm-eval RULER slice (niah_single_2, niah_multikey_1, ruler_vt; 25
samples/length; max_gen_toks=256 - the lm-eval default truncates chat
models' ruler_vt answers and invalidates the score).

| Metric | NVFP4 (tonyd2wild image, DFlash2 k=7) | EXL3 (Mia kit, DFlash2 k=7) |
| --- | --- | --- |
| Prefill pp2048 | 1792 +/- 89 tok/s | 940 +/- 76 tok/s |
| Decode tg128 | 52.5 +/- 0.5 (peak 66) | 49.0 +/- 1.3 (peak 59) |
| TTFT @2K | 1.15 s | 2.20 s |
| GSM8K 200q (flex/strict) | 89.0 / 87.5 % | 88.0 / 86.5 % |
| RULER 8K (s2/mk1/vt) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| RULER 16K | - | 1.0 / 1.0 / 1.0 |
| RULER 32K | 1.0 / 1.0 / - | - |
| RULER 64K (with topk mod) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| RULER 120-131K (with topk mod) | 1.0 / 1.0 / 1.0 @131072 | 1.0 / 0.92 / 0.63 @122880 |
| KV pool (boot log) | 3,895,606 tok | ~1.2M tok @131K window |

KV-pool note: the gap is allocation policy plus real runtime footprint,
not page cache - verified by booting EXL3 with a 5s cache flusher on all
nodes (pool unchanged: 1,197,617 vs 1,215,058 unflushed). NVFP4 pins
24 GiB/rank explicitly and runs eager (no graph memory); EXL3 KV is the
residue after vLLM profiles Mia's stack (EXL3 workspaces + CUDA graphs)
under her 0.87 util fraction. An explicit kv-cache-memory pin would grow
the EXL3 pool but is deliberately not applied (kept faithful to Mia's
config). Conversely the NVFP4 pin costs capacity: measured unpinned,
the profiler hands KV ~44 GiB/rank -> 7,193,816 tokens (1.85x the
pinned 3.9M; ~27 sessions @262K). The pin stays as the validated
default - removing it thins the UMA host-OOM margin tony's envelope
(docker --memory cap + flusher) exists to protect. Flip deliberately
if capacity is the priority.
| Max-context concurrency | 3.7 @1M / 14.8 @262K | 9.3 @131K |

Takeaways:
  - NVFP4 (tony's stack) beats its author's own 36 tok/s report (dual-rail
    fabric; his launcher wires one rail) and leads EXL3 on every speed
    metric at TP=4. Mia's 62.9 tok/s TP2 figure did not carry to TP4.
  - Quality is statistically identical across quants (GSM8K within
    stderr; RULER retrieval perfect on both).
  - Speed numbers are speculative-decode-dependent (DFlash2 k=7 both
    sides; acceptance varies with content) and say nothing about work
    quality beyond the GSM8K/RULER probes above.
  - Verdict: NVFP4 (tonyd2wild stack) wins speed, KV capacity, and
    extreme-length quality (perfect through 131K; EXL3's 4bpw degrades
    at 120K: vt 0.63, retrieval 0.92). EXL3's case is quality parity
    <=64K at 10% fewer weight bytes.
  - The early low ruler_vt scores (0.288/0.352) were a harness artifact:
    lm-eval's default generation budget truncates chat-model VT answers.
    With max_gen_toks=256 both stacks score 1.0 (through 131K on NVFP4).

### The GB10 32K ceiling and mods/fix-glm53-topk-sm120

Until 2026-08-30 every GLM-5.3-Flash stack on GB10 had an undocumented
hard ceiling: any request past ~32K tokens (sparse-indexer activation)
killed the engine in launch_persistent_topk (topk.cu:138) - the
persistent launch needs total_ctas <= num_sms*occupancy (48 on GB10 vs
77-90 required) and the FilteredTopK fallback needs 128KB smem/block
(GB10: 101KB, B200: 227KB). The NVFP4 stack boots at 1M and dies on the
first long request; the EXL3 stack dies at boot profiling for any
max_model_len that engages the sparse path. Nobody had published a real
>32K prompt run on GB10 - large advertised windows were boot-tested, not
serve-tested. Not a hardware limit of Spark itself: DeepSeek-V4-Flash
serves 800K on this same cluster via different kernels.

Root cause: the persistent_topk selector in sparse_attn_indexer.py /
sparse_attn_indexer_kpool.py lacks the sm120-family exclusion its
cooperative_topk sibling already has. mods/fix-glm53-topk-sm120 adds it,
routing GB10 to the generic top_k_per_row_decode kernel already present
in the else branch. Validated: EXL3 TP4 boots at 131072, serves 75/75
long requests, RULER 1.000/1.000/1.000 at 65,536 tokens - to our
knowledge the first >32K GLM-5.3-Flash serving on this hardware. At
122,880 tokens: 75/75 served, retrieval 1.0/0.92, ruler_vt 0.632 -
retrieval-grade quality holds through 120K; complex state tracking
degrades past 64K (model characteristic, not a serving failure). Both
recipes now carry the mod. Fallback-kernel long-context throughput is
lower than the persistent kernel would be on datacenter parts; measured
pace at 64K was ~52-60 s/item including prefill. Tracking: issue #27;
upstream (vLLM glm5_next) and image maintainers should receive this.

Session bug ledger (all fixed in recipes on this branch): missing
VLLM_MLA_NOPE_PAD_ROPE env gate (boot death in fp8_ds_mla cache); TP4
FlashInfer autotune wedge on the stock image (superseded by tonyd2wild
image switch); Glm5NextProcessor requires a local model path (HF id form
crashes); EXL3 multimodal warmup OOM on 121 GiB UMA (--language-model-only);
the 32K topk ceiling (mod above).
