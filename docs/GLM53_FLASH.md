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
