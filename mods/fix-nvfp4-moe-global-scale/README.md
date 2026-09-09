# fix-nvfp4-moe-global-scale

Fixes vLLM issue [#54150](https://github.com/vllm-project/vllm/issues/54150)
inside the container: the fused `[w1; w3]` NvFp4 MoE repack takes only w1's
per-expert global scale and merely logs when w3's differs
(`w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be
affected.`). Every w3 weight in such an expert is then dequantized with the
wrong global scale. On GLM-5.3-Flash NVFP4 (LibertAIDAI and all uncensored
re-exports, e.g. orcarouter) 68.5% of expert pairs differ, ratio up to 10x,
and the model emits invalid UTF-8 byte-token sequences (U+FFFD in Korean and
emoji text) at temperature 0.

`patch_nvfp4_moe_gscale.py` inserts one torch-only helper and rewrites the
single-gscale block in both NVFP4 MoE loaders:

| File | Scale semantics | Shared value |
| --- | --- | --- |
| `compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py` | `weight_global_scale`, divisor | per-expert **min** |
| `modelopt.py` | `weight_scale_2`, multiplier | per-expert **max** |

Each shard's ratio to the shared value is <= 1 by construction and is folded
into that shard's E4M3 block scales, so the dequantized weight is unchanged
and no clamping is needed. The only cost is one extra E4M3 rounding on the
moved shard's block scales. This is the same arithmetic as the validated
variants in the issue thread (8/6 -> 0/6 U+FFFD on 2x GB10 at our vLLM
commit) and mirrors vLLM's own FP8 MoE path
(`process_fp8_weight_tensor_strategy_moe`).

Idempotent and fail-closed: pinned anchors are preflighted, a drifted anchor
aborts the launch. Patches whichever loader files exist (`NVFP4_CT_MOE_PY`,
`NVFP4_MODELOPT_PY` override the paths for tests). Verified on vLLM
`0.1.dev20051+g487ecf187` (image `vllm-node-glm5.3-flash`).

Test: `tests/test_fix_nvfp4_moe_global_scale_mod.sh`. Investigation and
measurements: `docs/GLM53_FLASH.md`.
