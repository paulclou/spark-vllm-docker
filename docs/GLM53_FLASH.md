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
config). The NVFP4 pin was A/B'd unpinned (2026-08-30). An initial
run suggested unpinned degraded 131K retrieval, but that was a
measurement artifact (truncated generation budget - the same trap as
the ruler_vt scores). The protocol-matched retest (n=25, 256-token
budget) scored 1.00 / 0.96 / 1.00 at 131,072 tokens - statistically
identical to pinned - with the 1.85x larger auto pool (7.2M tokens,
~44 GiB/rank). Verdict: the pin has no measured benefit on this stack;
both fork recipes now run unpinned (auto). The Mia backports mod was
exonerated by the same A/B.
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

## The serve recipe (recipes/glm-5.3-flash-serve.yaml, 2026-08-30)

Production config, built ground-up as a replication of the official vLLM
recipe (recipes.vllm.ai GLM-5.3-Flash, GB200 NVL4 profile) with exactly
three contract exceptions: image (tonyd2wild sm121-v11-dflash2 - official
x86 image cannot run on aarch64), checkpoint (LibertAIDAI NVFP4 - official
offers RedHatAI NVFP4/FP8/BF16), TP x nodes (4 x 1-GPU Sparks vs official
8 x 2; --nnodes/--node-rank/--master-addr and NCCL/GLOO IFACE env are
generated per node by launch-cluster.sh). Doc priority for deviations:
vLLM official -> LibertAI -> Mia -> tony.

Deviations discovered mandatory by boot testing (~10 cycles):

- **gpu-memory-utilization 0.85**: this build's 0.92 default asks 111.95
  GiB; GB10 UMA has 110.19 of 121.69 GiB free at boot (host owns the
  rest) and the engine refuses to start. 0.85 is the tony-validated value
  every measured result used.
- **HF_HUB_OFFLINE=1** (universal, both tony's and Mia's images):
  Glm5NextProcessor open()s processor_config.json relative to the model
  argument; offline mode makes vLLM resolve the HF id to the local
  snapshot path first. Without it the API server crashes at boot.
- **Cache refs churn**: LibertAI pushed 3 README-only revisions in 24h
  (357b45cc full weights -> 436914c7 -> caca4e6a sparse); any online tool
  touching the repo moves the cache's refs/main pointer, and a boot then
  fails "cannot find weights" at the new snapshot. Fix: repin refs/main
  to 357b45cc; container (root) writes can leave refs root-owned - chown
  1000:1000. An explicit --revision pin was proposed and declined
  (owner decision, 2026-08-30); refs hygiene is the standing mitigation.

Approved tuning deviations from vLLM defaults (owner-approved, all
smoke-validated together):

- max-num-seqs 16 (default 128) - fleet seat cap consistent with ds4f-1m.
- max-num-batched-tokens 8192 (default 2048) - 4x long-prompt prefill.
- block-size 2304 (default 16) - 18x128, aligned to the sparse indexer's
  128-token tiles; both validated GB10 deployments chose it
  independently; best 72K needle of the campaign (36.6s vs 42s prior).
- max-model-len 1048576 explicit - same shortfall-becomes-boot-failure
  policy as ds4f-1m. Validated to 131,072-token prompts; beyond is
  configured but untested.
- DFlash2 k=7 explicit probabilistic+standard (52.5 vs MTP-5's ~38
  tok/s). Drafter incoai/GLM-5.3-Flash-DFlash2 is CC BY-NC-ND
  (non-commercial); official MTP-5 config is the commercial fallback.
  Drafter must be in every node's HF cache.
- --enable-prefix-caching stated explicitly (vLLM default is on).
- KV pin: none (auto pool 6.69M tokens @0.85). The 24 GiB/rank pin was
  A/B'd and has no measured benefit (see KV-pool note above).

Thinking/parser findings (probed on the live endpoint):

- Thinking is ALWAYS ON - the template has no enable_thinking kwarg.
  Clients dial it with chat_template_kwargs {"reasoning_effort":
  "low"|"high"|"max"} (default max; measured: low ~50 think tokens/1.5s,
  max ~310/7.8s). clear_thinking=true (zai chat recommendation) strips
  prior turns' think blocks from the prompt; set false only for
  benchmark repro/debugging of multi-turn reasoning.
- Reasoning arrives in message.reasoning (NOT reasoning_content) in this
  vLLM build. An early probe read the wrong field and blamed the glm45
  parser for discarding think blocks - glm45 (the official recipe's
  choice) is not disproven. deepseek_r1 is kept per the LibertAI model
  card and is probe-verified working.

Smoke status (final config, 2026-08-30): boots ~15 min; 72,218-token
needle exact in 36.6s; reasoning present; effort dial works; glm47 tool
calls structured correctly; KV pool 6,692,504 tokens (~6.4 full 1M
sessions, ~25 @262K, 16-seat cap).

### Serve-config formal campaign (2026-08-30, protocol matched to the
### bench campaign above)

| Metric | Serve config | Bench reference |
| --- | --- | --- |
| Decode tg128 | 65.2 +/- 5.1 tok/s (peak 77.3) | 52.5 +/- 0.5 |
| Prefill pp2048 | 1501 +/- 64 tok/s | 1792 +/- 89 |
| TTFT @2K | 1.37 s | 1.15 s |
| GSM8K 200q (flex/strict) | 90.0 / 88.0 % | 89.0 / 87.5 % |
| RULER s2/mk1 8K-131K | 1.0 / 1.0 at every length | same |
| ruler_vt | 1.0 @8K/32K; 0.88 @64K; 0.736 @131K* | 1.00 @64K/131K |

Decode is +24% over the bench config (CUDA graphs on vs eager; block
2304). The prefill/TTFT cost (-16% / +0.2s) is real (outside error bars)
and accepted: decode dominates agent-serving wall-clock.

*The ruler_vt depth "degradation" is a HARNESS ARTIFACT, diagnosed
2026-08-30 with logged samples: max_gen_toks=256 counts thinking tokens,
the model reasons longer at depth, and the deepseek_r1 parser correctly
strips reasoning out of content - so deep-context answers get truncated
(one sample: content cut mid-sentence before the variable list; another:
0 content tokens, all 256 spent thinking). Same 8 prompts @131K: budget
256 scores 0.75, budget 1024 scores 1.00. The bench config "passed" at
256 only because it serves without a reasoning parser, so think-text
leaks into content and RULER's string match finds the answer inside the
reasoning. Rule for ALL evals against endpoints with a reasoning parser:
budget must cover thinking + answer (max_gen_toks >= 1024 for RULER), a
stricter form of the max_gen_toks trap already in this file.

MM inference VALIDATED on the serve config (2026-08-30 smoke probes,
tony image, no --language-model-only): a shapes/colors/text image and a
bar chart both described exactly (all shapes, positions, colors, the
rendered word, chart values and the odd-colored bar) in 1.5-2.8 s at
~300-385 prompt tokens. The vision front-end costs no extra flags on
this stack; probe scripts in the session scratchpad.

Still unmeasured on the serve config: >131K prompts, DFlash2 acceptance
under 16-seat concurrency, MM under load/large images (probes were
smoke-grade).
