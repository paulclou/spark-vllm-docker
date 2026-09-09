#!/usr/bin/env bash
# Berkeley Function-Calling Leaderboard (BFCL v4) against a live vLLM recipe.
#
# WHY: the tool-call path -- native `tools` on /v1/chat/completions parsed
# server-side by the recipe's tool-call parser (glm47, hermes, deepseek_v4) --
# is exercised by NEITHER llama-benchy (speed) nor lm-eval (gsm8k/ruler). BFCL
# is the public, comparable measure of whether the served model emits correct,
# parseable function calls. Run it after any image, recipe, or checkpoint
# change that could touch the tool-call contract.
#
# It talks to the server over the OpenAI-compatible API only (native function
# calling), so unlike lm-eval/llama-benchy it needs NO local tokenizer and no
# model download. Run it on the head node (127.0.0.1:8000 always works) or
# anywhere that can reach the tailnet URL (set BASE_URL). VLLM_API_KEY must be
# exported; it is the only place the key is read from.
#
# CAVEAT: BFCL prompts are short (<~4K tokens). This scores general tool-call
# correctness, comparable to public GLM/Qwen numbers -- it does NOT probe the
# long-context tool-call boundary. Keep that a separate probe.
#
# Everything the script installs is EPHEMERAL and lives under WORKDIR (a tmp
# dir, never $HOME). Results (BFCL's own JSON + score CSVs) land in OUTDIR and
# are NOT git-tracked; transcribe the headline numbers into the recipe's docs
# page per docs/BENCHMARKING.md.
#
# Usage:
#   tools/bfcl-bench.sh <served-model-name> [category ...]
#   tools/bfcl-bench.sh glm-5.3-flash-uncensored-nvfp4 single_turn
#   BASE_URL=https://<node>.<tailnet>.ts.net:8000/v1 \
#     tools/bfcl-bench.sh glm-5.3-flash-uncensored-nvfp4 single_turn multi_turn
#
# Categories: any BFCL collection or leaf (single_turn, multi_turn, live,
#   irrelevance, simple_python, multiple, parallel, multi_turn_long_context...).
#   Default: single_turn.
set -uo pipefail

MODEL="${1:?usage: bfcl-bench.sh <served-model-name> [category ...]}"; shift || true
CATS=("$@"); [ ${#CATS[@]} -eq 0 ] && CATS=(single_turn)

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
THREADS="${THREADS:-8}"
BFCL_VERSION="${BFCL_VERSION:-2026.3.23}"
WORKDIR="${WORKDIR:-${TMPDIR:-/tmp}/bfcl-harness}"
OUTDIR="${OUTDIR:-$WORKDIR/result-$MODEL}"
VENV="$WORKDIR/venv-$BFCL_VERSION"
REG_KEY="${MODEL}-FC"          # BFCL registry key == the --model argument

# --- API key: always VLLM_API_KEY from the environment (docs/BENCHMARKING.md).
#     No file or container lookup; never echo it.
KEY="${VLLM_API_KEY:-}"
if [ -z "$KEY" ]; then
  echo "ERROR: VLLM_API_KEY is not set; export it before running." >&2
  exit 1
fi

# --- Reachability check before the (slow) install.
code="$(curl -s -m 8 -o /dev/null -w '%{http_code}' "$BASE_URL/models" \
        -H "Authorization: Bearer $KEY" 2>/dev/null || echo 000)"
if [ "$code" != 200 ]; then
  echo "ERROR: $BASE_URL/models returned $code (000=unreachable, 401=bad key," >&2
  echo "       'HTTP request to an HTTPS server'=wrong scheme). Fix BASE_URL/key." >&2
  exit 1
fi

command -v uv >/dev/null || { echo "ERROR: uv not found (astral.sh/uv)." >&2; exit 1; }
mkdir -p "$WORKDIR" "$OUTDIR"

# --- Ephemeral, version-pinned venv (reused across runs of the same version).
if [ ! -x "$VENV/bin/bfcl" ]; then
  echo ">> installing bfcl-eval==$BFCL_VERSION into $VENV"
  uv venv "$VENV" >/dev/null
  # soundfile is an unpinned transitive import of qwen_agent that BFCL loads at
  # registry import time; without it every command dies on ModuleNotFoundError.
  uv pip install -p "$VENV/bin/python" "bfcl-eval==$BFCL_VERSION" soundfile >/dev/null
fi

# --- Register the served model as a generic OpenAI function-calling model.
#     The OpenAICompletionsHandler reads OPENAI_BASE_URL / OPENAI_API_KEY and
#     hits native /v1/chat/completions tools -- the same path clients use.
#     Idempotent: guarded by a marker so a reused venv is not appended twice.
CFG="$("$VENV/bin/python" -c 'import bfcl_eval.constants.model_config as m; print(m.__file__)')"
MARKER="# bfcl-bench.sh registered: $REG_KEY"
if ! grep -qF "$MARKER" "$CFG"; then
  cat >> "$CFG" <<PYEOF

$MARKER
MODEL_CONFIG_MAPPING["$REG_KEY"] = ModelConfig(
    model_name="$MODEL",
    display_name="$MODEL (vLLM, native FC)",
    url="local",
    org="local",
    license="unknown",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
)
PYEOF
fi

export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="$KEY"
export BFCL_PROJECT_ROOT="$OUTDIR"

echo ">> model=$MODEL  base=$BASE_URL  categories=${CATS[*]}  out=$OUTDIR"
for cat in "${CATS[@]}"; do
  echo ">> [generate] $cat"
  "$VENV/bin/bfcl" generate --model "$REG_KEY" --test-category "$cat" \
    --num-threads "$THREADS" || { echo "generate failed for $cat" >&2; continue; }
  echo ">> [evaluate] $cat"
  "$VENV/bin/bfcl" evaluate --model "$REG_KEY" --test-category "$cat" \
    || echo "evaluate failed for $cat" >&2
done

echo
echo ">> score CSVs under $OUTDIR/score/ :"
find "$OUTDIR/score" -name '*.csv' 2>/dev/null | sort
echo ">> transcribe the headline accuracy into the recipe's docs page"
echo "   (docs/GLM53_FLASH.md etc.), with this exact invocation, per"
echo "   docs/BENCHMARKING.md's provenance rule."
