#!/bin/bash
# vLLM #54150: fused NvFp4 MoE takes one global scale per expert but drops
# w3's when it differs from w1's. Requantize the block scales onto a shared
# per-expert global scale instead (both loaders). Fails closed on anchor drift.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/patch_nvfp4_moe_gscale.py"
