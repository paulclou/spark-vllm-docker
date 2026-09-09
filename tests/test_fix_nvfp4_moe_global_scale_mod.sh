#!/bin/bash
# mods/fix-nvfp4-moe-global-scale: patch both NvFp4 MoE loaders on fixtures,
# check idempotency and anchor-drift refusal, and verify the requantization
# arithmetic numerically (torch required for that last part; skipped if absent).
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD_DIR="$PROJECT_DIR/mods/fix-nvfp4-moe-global-scale"
MOD="$MOD_DIR/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

CT="$TMP_DIR/compressed_tensors_moe_w4a4_nvfp4.py"
MO="$TMP_DIR/modelopt.py"

# Fixtures: the pinned anchors verbatim, inside a syntactically valid module.
cat > "$CT" <<'PY'
import torch
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsW4A4MoeMethod:
    def process_weights_after_loading(self, layer) -> None:
        layer.w13_weight = layer.w13_weight_packed
        delattr(layer, "w13_weight_packed")

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.",
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

        return w13_weight_global_scale
PY

cat > "$MO" <<'PY'
import torch
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class ModelOptNvFp4FusedMoE:
    def process_weights_after_loading(self, layer) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()

        return w13_weight_scale_2
PY

run_mod() {
    NVFP4_CT_MOE_PY="$CT" NVFP4_MODELOPT_PY="$MO" bash "$MOD"
}

echo "== first application"
out=$(run_mod)
grep -Fq 'compressed_tensors_moe_w4a4_nvfp4.py: nvfp4-moe-gscale patched' <<<"$out"
grep -Fq 'modelopt.py: nvfp4-moe-gscale patched' <<<"$out"
for f in "$CT" "$MO"; do
    grep -Fq 'def _share_w13_global_scale(' "$f"
    grep -Fq 'replace_parameter(layer, "w13_weight_scale", w13_scale_shared)' "$f"
    ! grep -Fq 'Accuracy may be affected' "$f"
    [ "$(grep -c '# \[nvfp4-moe-gscale\]' "$f")" -eq 2 ]
    python3 -m py_compile "$f"
done
grep -Fq 'divisor=True' "$CT"
grep -Fq 'divisor=False' "$MO"

echo "== idempotent second application"
out=$(run_mod)
grep -Fq 'compressed_tensors_moe_w4a4_nvfp4.py: nvfp4-moe-gscale already patched' <<<"$out"
grep -Fq 'modelopt.py: nvfp4-moe-gscale already patched' <<<"$out"

echo "== anchor drift is refused"
DRIFT="$TMP_DIR/drift.py"
sed 's/Accuracy may be affected/accuracy may differ/' > "$DRIFT" <<'PY'
import torch
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class M:
    def f(self, layer):
        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].contiguous()
PY
if NVFP4_CT_MOE_PY="$TMP_DIR/absent.py" NVFP4_MODELOPT_PY="$DRIFT" bash "$MOD" >/dev/null 2>&1; then
    echo "FAIL: drifted anchor was accepted" >&2
    exit 1
fi
grep -Fq 'accuracy may differ' "$DRIFT"   # untouched

echo "== no loader file at all is refused"
if NVFP4_CT_MOE_PY="$TMP_DIR/absent.py" NVFP4_MODELOPT_PY="$TMP_DIR/absent2.py" bash "$MOD" >/dev/null 2>&1; then
    echo "FAIL: mod succeeded with no target" >&2
    exit 1
fi

echo "== requantization arithmetic"
if ! python3 -c 'import torch' 2>/dev/null; then
    echo "SKIP: torch not importable; numeric check not run"
    exit 0
fi
python3 - "$MOD_DIR/patch_nvfp4_moe_gscale.py" <<'PY'
import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("patch_mod", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

# Exec the helper exactly as it is inserted into the loader files.
ns = {"torch": torch, "init_logger": lambda name: None}
exec(mod.HELPER, ns)
share = ns["_share_w13_global_scale"]

torch.manual_seed(0)
E, SHARDS, N, KG = 5, 2, 6, 4
fp8 = torch.float8_e4m3fn


def dequant_scale(bs, gs, divisor):
    """Effective per-block multiplier, shard-wise: block_scale (/ or *) gs."""
    b = bs.to(torch.float32).view(E, SHARDS, N, KG)
    g = gs.to(torch.float32).view(E, SHARDS, 1, 1)
    return b / g if divisor else b * g


for divisor in (True, False):
    # Block scales spanning the E4M3 normal range; global scales with equal,
    # mild (7%) and extreme (10x) w1/w3 mismatches, like the GLM checkpoint.
    bs = (torch.rand(E, SHARDS * N, KG) * 60 + 0.5).to(fp8)
    base = torch.tensor([21504.0, 100.0, 3.0e-5, 7.0, 1.0]) if divisor else torch.tensor([4.65e-5, 1.0, 0.02, 300.0, 1.0])
    ratios = torch.tensor([1.2321, 1.0, 10.0, 0.5, 1.0])
    gs = torch.stack([base, base * ratios], dim=1)

    new_bs, shared = share(bs, gs, divisor=divisor)

    assert new_bs.dtype == fp8 and new_bs.shape == bs.shape
    assert shared.shape == (E,) and shared.is_contiguous()
    expected_shared = gs.min(dim=1).values if divisor else gs.max(dim=1).values
    assert torch.equal(shared, expected_shared), (shared, expected_shared)

    # Ratios <= 1: no rescaled block scale may exceed its original.
    assert torch.all(new_bs.to(torch.float32) <= bs.to(torch.float32) * (1 + 1e-6))

    # Dequantized weight is preserved up to one E4M3 rounding (half-ulp of
    # a 3-bit mantissa is 6.25%); the exact-float requant must round-trip.
    before = dequant_scale(bs, gs, divisor)
    after = dequant_scale(new_bs, shared.unsqueeze(1).expand(E, SHARDS), divisor)
    rel = ((after - before).abs() / before).max().item()
    assert rel <= 0.0625 + 1e-6, f"divisor={divisor}: rel err {rel}"
    exact = dequant_scale(
        (bs.to(torch.float32).view(E, SHARDS, N, KG)
         * ((shared.unsqueeze(1) / gs) if divisor else (gs / shared.unsqueeze(1))).view(E, SHARDS, 1, 1)
         ).view(E, SHARDS * N, KG),
        shared.unsqueeze(1).expand(E, SHARDS), divisor)
    assert torch.allclose(exact, before, rtol=1e-5), "exact requant must be lossless"

    # Experts whose scales already match are left bit-identical.
    equal = ratios == 1.0
    assert torch.equal(new_bs.view(E, -1).float()[equal], bs.view(E, -1).float()[equal])

    # The unpatched behaviour (w1's scale for both shards) is what we are
    # fixing: on the 10x expert w3 comes out 10x too large (divisor) or 90%
    # too small (multiplier); the fix is within one E4M3 rounding.
    naive = dequant_scale(bs, gs[:, :1].expand(E, SHARDS), divisor)
    naive_rel = ((naive - before).abs() / before).view(E, -1).max(dim=1).values
    assert naive_rel[2] > (8.0 if divisor else 0.85), naive_rel
    print(f"divisor={divisor}: max rel err after fix {rel:.4f}; unpatched w3 error on the 10x expert {naive_rel[2]:.2f}")
print("requantization arithmetic OK")
PY
echo "PASS"
