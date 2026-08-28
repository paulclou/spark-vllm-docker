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

## DFlash2: the next speed step (watch item, not yet runnable here)

DFlash2 is z-lab's block-diffusion drafter (drafts a whole block per pass;
2.4-2.8x measured on GLM-5.3-Flash at TP=4 on GB300s). A ready drafter
exists: incoai/GLM-5.3-Flash-DFlash2. Blockers for this cluster, checked
2026-08-28:
  1. Engine support is SGLang-only via unmerged PR sgl-project/sglang#36708
     (vLLM support announced, not shipped).
  2. The draft path requires the fa4 attention backend - the same
     CuTe-DSL kernel family that fails MLIR compilation on sm_121 (what the
     qwen38fn Triton-fallback patch works around). A GB10 run needs an
     equivalent fallback for the drafter.
Revisit when the SGLang PR merges or a GB10 deployment is published; the
expected payoff is ~25 -> 40-50 tok/s single-stream, which would also reopen
the engine question for GLM.
