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
