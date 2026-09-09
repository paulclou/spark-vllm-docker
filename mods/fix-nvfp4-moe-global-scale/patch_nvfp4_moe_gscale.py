#!/usr/bin/env python3
"""Requantize NVFP4 MoE block scales onto one shared w1/w3 global scale.

vLLM fuses each expert's gate (w1) and up (w3) projections into one
``[w1; w3]`` NvFp4 GEMM whose kernels accept ONE global scale per expert.
Both NVFP4 MoE loaders (compressed-tensors and ModelOpt) resolve that by
taking w1's global scale and merely logging when w3's differs::

    w13_weight_global_scale = layer.w13_weight_global_scale[:, 0]

Checkpoints whose experts were quantized as separate gate_proj/up_proj
modules with per-tensor amax carry different w1/w3 global scales (GLM-5.3-Flash
NVFP4 from LibertAIDAI and every uncensored re-export of it: 68.5% of expert
pairs differ, ratio up to 10x).  Every w3 weight in such an expert is then
dequantized with the wrong global scale, and the model emits invalid UTF-8
byte-token sequences (U+FFFD in Korean/emoji text) at temperature 0.  This is
vLLM issue #54150; the FP8 MoE path already handles the same situation
correctly (``process_fp8_weight_tensor_strategy_moe`` requantizes onto the
max), the NVFP4 loaders never got the equivalent.

The fix: pick one per-expert global scale and fold each shard's ratio into
its E4M3 block scales, so ``q * block_scale * scale_2`` (ModelOpt, multiplier)
or ``q * block_scale / global_scale`` (compressed-tensors, divisor) is
unchanged.  The shared value is chosen so every ratio is <= 1 (max of the
multipliers, min of the divisors): no clamping is ever needed, and the only
cost is one extra E4M3 rounding on the moved shard's block scales.  Same
arithmetic as the validated variants in the #54150 thread.

Fail-closed, idempotent, preflights the pinned anchors before writing.
Patches whichever of the two loader files exist; refuses if neither does.
Target paths can be overridden for tests with ``NVFP4_CT_MOE_PY`` and
``NVFP4_MODELOPT_PY``.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import NamedTuple

SITE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization"

MARK = "# [nvfp4-moe-gscale]"

LOGGER_ANCHOR = "\nlogger = init_logger(__name__)\n"

# Inserted once per file, right after the module logger. Pure torch; no vLLM
# imports so the test suite can exec it standalone.
HELPER = '''
logger = init_logger(__name__)


# [nvfp4-moe-gscale] vLLM #54150. The fused [w1; w3] NvFp4 MoE kernel takes
# one global scale per expert, but checkpoints quantized with separate
# gate_proj/up_proj carry different w1/w3 global scales. Picking w1's
# mis-scales every w3 weight by the ratio (up to 10x on GLM-5.3-Flash NVFP4).
# Requantize instead: share one per-expert global scale and fold each shard's
# ratio (<= 1 by construction, so no clamping) into its E4M3 block scales.
# Mirrors the FP8 MoE path (process_fp8_weight_tensor_strategy_moe).
def _share_w13_global_scale(
    block_scale: torch.Tensor, gs_pair: torch.Tensor, *, divisor: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rescaled block scales, shared per-expert global scale).

    block_scale: (E, num_shards * N, K // group) E4M3 block scales with the
        shards stacked along dim 1 in shard order (w1 rows, then w3 rows).
    gs_pair: (E, num_shards) per-shard global scales.
    divisor: True when dequant is ``q * block_scale / gs`` (compressed-tensors
        ``weight_global_scale``); False when it is ``q * block_scale * gs``
        (ModelOpt ``weight_scale_2``).
    """
    gs = gs_pair.to(torch.float32)
    num_experts, num_shards = gs.shape
    _, n2, kg = block_scale.shape
    if divisor:
        shared = gs.min(dim=1).values
        ratio = shared.unsqueeze(1) / gs
    else:
        shared = gs.max(dim=1).values
        ratio = gs / shared.unsqueeze(1)
    bs = block_scale.to(torch.float32).view(
        num_experts, num_shards, n2 // num_shards, kg
    )
    bs = bs * ratio.view(num_experts, num_shards, 1, 1)
    return bs.view(num_experts, n2, kg).to(block_scale.dtype), shared.contiguous()
'''


class Target(NamedTuple):
    env: str
    default: Path
    anchor: str
    patched: str
    label: str


CT_ANCHOR = '''        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.",
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()
'''

CT_PATCHED = '''        # Use a single gscale for w13.
        # [nvfp4-moe-gscale] requantize instead of dropping w3's scale
        # (see _share_w13_global_scale; vLLM #54150).
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale != w3_weight_global_scale; requantizing "
                "the block scales onto a shared per-expert global scale "
                "(vLLM #54150)."
            )
            w13_scale_shared, w13_weight_global_scale = _share_w13_global_scale(
                layer.w13_weight_scale.data,
                layer.w13_weight_global_scale.data,
                divisor=True,
            )
            replace_parameter(layer, "w13_weight_scale", w13_scale_shared)
        else:
            w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()
'''

MO_ANCHOR = '''        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()
'''

MO_PATCHED = '''        # Use a single gscale for w13.
        # [nvfp4-moe-gscale] requantize instead of dropping w3's scale
        # (see _share_w13_global_scale; vLLM #54150).
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 != w3_weight_scale_2; requantizing the block "
                "scales onto a shared per-expert global scale (vLLM #54150)."
            )
            w13_scale_shared, w13_weight_scale_2 = _share_w13_global_scale(
                layer.w13_weight_scale.data,
                layer.w13_weight_scale_2.data,
                divisor=False,
            )
            replace_parameter(layer, "w13_weight_scale", w13_scale_shared)
        else:
            w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()
'''

TARGETS = (
    Target(
        env="NVFP4_CT_MOE_PY",
        default=Path(
            f"{SITE}/compressed_tensors/compressed_tensors_moe/"
            "compressed_tensors_moe_w4a4_nvfp4.py"
        ),
        anchor=CT_ANCHOR,
        patched=CT_PATCHED,
        label="compressed-tensors NvFp4 MoE",
    ),
    Target(
        env="NVFP4_MODELOPT_PY",
        default=Path(f"{SITE}/modelopt.py"),
        anchor=MO_ANCHOR,
        patched=MO_PATCHED,
        label="ModelOpt NvFp4 MoE",
    ),
)

REQUIRED_IMPORT = (
    "from vllm.model_executor.utils import replace_parameter, set_weight_attrs\n"
)


def verified_state(text: str, target: Target) -> bool:
    return (
        text.count(target.anchor) == 0
        and text.count(target.patched) == 1
        and text.count(HELPER) == 1
        and text.count("def _share_w13_global_scale(") == 1
    )


def prepare(source: str, target: Target) -> tuple[str, str]:
    marks = source.count(MARK)
    if marks:
        if not verified_state(source, target):
            raise ValueError(
                f"partial/inconsistent nvfp4-moe-gscale patch (markers={marks})"
            )
        return source, "already patched"
    if source.count(target.anchor) != 1:
        raise ValueError(
            f"pinned single-gscale anchor drifted (found {source.count(target.anchor)})"
        )
    if source.count(LOGGER_ANCHOR) != 1:
        raise ValueError(
            f"pinned logger anchor drifted (found {source.count(LOGGER_ANCHOR)})"
        )
    if REQUIRED_IMPORT not in source:
        raise ValueError("replace_parameter import not found; re-derive the patch")
    if "_share_w13_global_scale" in source:
        raise ValueError("helper name already present without marker; re-derive")
    patched = source.replace(LOGGER_ANCHOR, HELPER, 1)
    patched = patched.replace(target.anchor, target.patched, 1)
    if not verified_state(patched, target):
        raise ValueError("post-patch verification failed")
    return patched, "patched"


def replace_file(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.nvfp4-moe-gscale.tmp")
    try:
        tmp.write_text(text)
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}*.pyc"):
            pyc.unlink(missing_ok=True)


def main() -> int:
    handled = 0
    for target in TARGETS:
        path = Path(os.environ.get(target.env, target.default))
        if not path.is_file():
            print(f"{target.label}: {path} not present, skipped")
            continue
        source = path.read_text()
        try:
            patched, action = prepare(source, target)
        except ValueError as exc:
            raise SystemExit(f"{target.label} preflight failed: {exc}") from exc
        compile(patched, str(path), "exec")
        if patched != source:
            replace_file(path, patched)
            clear_pyc(path)
        print(f"{path.name}: nvfp4-moe-gscale {action}")
        handled += 1
    if not handled:
        raise SystemExit("nvfp4-moe-gscale: no NvFp4 MoE loader file found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
