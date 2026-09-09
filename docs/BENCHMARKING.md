# Benchmarking vLLM recipes on the Spark cluster

Model-agnostic mechanics for measuring a live recipe: TLS trust, the
llama-benchy / lm-eval / spec-decode invocations, and the harness
landmines that cost real time. Per-recipe results, tuning history, and
model-specific deviations live in that recipe's own docs page (e.g.
`docs/GLM53_FLASH.md`), which references this file rather than repeating
it. Everything here applies equally to the qwen, ds4f, and GLM lanes.

## TLS: use the tailnet URL

The `vllm@` units serve plain HTTP on the head node; `tailscale serve` in
front of them presents a publicly trusted certificate for the node's
tailnet name. Point every client at
`https://<node>.<tailnet>.ts.net:8000/v1` and no custom trust bundle is
needed. Never point `SSL_CERT_FILE` at a lone CA file - it replaces the
system bundle and breaks uv's own PyPI fetches. `tools/bench-serving.py`
uses `CERT_NONE` and the probe scripts take `--insecure`; neither is
required against the tailnet URL.

The API key is always `VLLM_API_KEY`, exported in the shell that runs the
tool. Every script and command below reads it from the environment and
nowhere else - no key file, no `docker exec`, no ssh lookup. Never echo it
into a shared log.

## Host setup

uv/uvx via the standard `astral.sh/uv/install.sh` (installs to
`~/.local/bin`). A cluster node is a fine bench host and already has the
model snapshot in its HF cache; the workstation often cannot reach
huggingface.co, so run evals on a node and pass LOCAL paths.

## Speed - llama-benchy

```bash
SNAP=$(ls -d ~/.cache/huggingface/hub/models--<ORG>--<MODEL>/snapshots/*/)
uvx llama-benchy@0.4.0 \
  --base-url https://<node>.<tailnet>.ts.net:8000/v1 --api-key "$VLLM_API_KEY" \
  --model <served-name> --tokenizer "$SNAP" \
  --pp 2048 --tg 128 --runs 3 \
  --save-result ~/bench.json --format json
```

- `--base-url` MUST include the `/v1` suffix (404 without it).
- `--tokenizer` MUST be the local snapshot path - the bare HF id triggers
  a re-download that fails offline.
- Decode tok/s is noisy (CV ~13% on GB10); for a stable number or an A/B,
  raise `--runs` (15+) and use `--tg 256 --exact-tg`. Runs are seconds
  each - the engine boot is the real cost, so over-sample runs freely.

## Quality - lm-eval

Dependencies: `--from "lm_eval[api]" --with transformers` is the minimum
(the `[api]` extra does NOT pull transformers; the run dies on import
without it). For RULER add `,ruler` to the extra plus
`--with wonderwords --with nltk`.

```bash
OPENAI_API_KEY="$VLLM_API_KEY" \
uvx --from "lm_eval[api]" --with transformers lm_eval \
  --model local-completions \
  --model_args "model=<served-name>,base_url=https://<node>.<tailnet>.ts.net:8000/v1/completions,num_concurrent=8,max_retries=3,tokenizer=$SNAP,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --limit 200 \
  --output_path ~/lm-eval-results/gsm8k
```

RULER: `--tasks niah_single_2,niah_multikey_1,ruler_vt`, one length via
`--metadata '{"max_seq_lengths":[<LEN>]}'`, **`max_length=1050000` in
`--model_args`** (mandatory, see the first landmine below), and
**`--gen_kwargs max_gen_toks=256`** (mandatory - the default budget
truncates chat-model ruler_vt answers and invalidates the score). 25
samples/length (`--limit 25`).

### lm-eval landmines (each cost real time, 2026-08-31 and 2026-09-07)

- **Pass `max_length=<served context>` in `--model_args` for anything
  longer than ~1.8K tokens.** `local-completions` defaults to
  `max_length=2048` and silently LEFT-TRUNCATES every prompt to
  `max_length - 1 - max_gen_toks` tokens before sending it (lm-eval
  0.4.13, `api_models.py`). No warning is printed. RULER 8K then scores
  0.16-0.32 and 64K scores ~0 - the needle is cut off and the model answers
  with whatever number is left in the tail of the essay - which looks
  exactly like a catastrophic model regression. The 2026-08-31 runs used
  `max_length=1050000`; this line was missing from the invocation above
  until 2026-09-07 and cost an hour of false-alarm debugging against a
  freshly patched server. If RULER drops while GSM8K holds, check this
  first: replay one failing prompt by hand through `/v1/completions`.
- **Validate the request count before trusting the table.** A partial or
  dropped run looks like a clean pass. Confirm fired ==
  tasks x lengths x limit (e.g. 3 x 1 x 25 = 75).
- **One length per RULER invocation.** A multi-value
  `max_seq_lengths:[65536,131072]` silently ran only the FIRST length
  (75/75, one length; the second absent from the table, no sentinel row).
- **Long context (>=64K): `num_concurrent<=2` AND `timeout=3600`.**
  lm-eval's aiohttp session timeout defaults to 300s. At high concurrency
  the multi-minute long-context prefills queue and later requests wait
  past 300s, so the session tears down with `RuntimeError: Session is
  closed` - a CLIENT-side failure; the server never receives the requests
  (its queue stays 0). At concurrency 1-2 each request finishes inside the
  raised budget. (Short-context tasks like GSM8K are fine at
  num_concurrent=8.)
- **Never pipe an lm-eval run through `head`.** `... | grep X | head -N`
  sends SIGPIPE up the chain when head closes and kills the eval
  mid-run - it exits 0 (head's status) with no results written. Use
  `tail`, or redirect to a file.
- **RULER long-context build is a CPU-bound false-stall.** Context
  construction is single-threaded token-by-token noise insertion (one
  core ~100%, minutes); the GPU/server queue sits at 0 the whole time.
  Do not mistake the idle server for a hang.

## Spec-decode acceptance (free, from /metrics)

After a few varied requests, read off `/metrics`:
`sum(vllm:spec_decode_num_accepted_tokens_total) /
vllm:spec_decode_num_drafts_total` gives tokens-per-step (+1 bonus token
per draft). `vllm:spec_decode_num_accepted_tokens_per_pos_total{position=N}`
shows the per-depth acceptance decay - the lever for tuning
`num_speculative_tokens` (k): if deep positions rarely accept, a lower k
wastes less draft compute. A fresh engine boot zeroes these counters, so
one boot per variant gives a clean per-variant reading.

## Tool calling - BFCL (`tools/bfcl-bench.sh`)

Neither llama-benchy (speed) nor lm-eval (gsm8k/ruler) exercises the
tool-call path: native `tools` on `/v1/chat/completions` parsed server-side
by the recipe's `--tool-call-parser` (glm47, hermes, deepseek_v4). The
Berkeley Function-Calling Leaderboard (BFCL v4) is the public, comparable
measure of that path. Run it after any image, recipe, or checkpoint change
that could touch the tool-call contract.

```bash
# on the head node (127.0.0.1:8000 always works); VLLM_API_KEY must be exported
tools/bfcl-bench.sh glm-5.3-flash-uncensored-nvfp4 single_turn
# or against the tailnet URL from anywhere
BASE_URL=https://<node>.<tailnet>.ts.net:8000/v1 \
  tools/bfcl-bench.sh glm-5.3-flash-uncensored-nvfp4 single_turn multi_turn live
```

- Talks the OpenAI API only (native FC), so - unlike lm-eval/llama-benchy -
  it needs NO local tokenizer and no model download. The generic
  `OpenAICompletionsHandler` is used deliberately; BFCL's OSS-model path
  formats prompts client-side and would bypass the server's tool parser.
- The script installs a version-pinned bfcl-eval into a tmp venv (never
  `$HOME`), registers the served model as a generic OpenAI FC model, checks
  reachability + key before the slow install, and runs generate + evaluate.
- `soundfile` is pinned alongside bfcl-eval: it is an unpinned transitive
  import of `qwen_agent` that BFCL loads at registry-import time, and the CLI
  dies on `ModuleNotFoundError` without it.
- API key: `VLLM_API_KEY` from the environment, nothing else. The script
  exits early if it is unset.
- Concurrency: vLLM caps running requests (16 on the GLM recipes), so
  `THREADS` above that cap only queues. `single_turn` (3,401 cases) took
  ~60 min at 16 streams with reasoning on (mean latency ~36 s, p95 ~86 s).
  A killed run resumes: existing result ids are skipped.
- Scan the raw responses, not just the score: BFCL's accuracy hides
  garble. Grep `result/**/*.json` for U+FFFD, `<tool_call>` inside string
  results, CJK ratio, and long repeats (see docs/GLM53_FLASH.md for what
  this found). `--run-ids` with a `test_case_ids_to_generate.json` under a
  fresh `BFCL_PROJECT_ROOT` reruns just the flagged ids.
- The CSV "Overall Acc" averages every BFCL category, counting unrun ones
  (multi_turn, agentic) as zero. Read `data_non_live.csv` and
  `data_live.csv` for the categories you actually ran.
- Categories: any BFCL collection (`single_turn`, `multi_turn`, `live`) or
  leaf (`simple_python`, `multiple`, `parallel`, `irrelevance`,
  `multi_turn_long_context`). Results (JSON + score CSVs) land in `OUTDIR`
  (a tmp dir), NOT git - transcribe the headline accuracy into the recipe's
  docs page per the provenance rule below.

**CAVEAT: BFCL prompts are short (<~4K tokens).** This scores general
tool-call correctness, comparable to public GLM/Qwen numbers. It does NOT
probe the long-context tool-call boundary (where this cluster's episodic
garble lives, ctx >~150K); keep that a separate probe.

## Refusal / abliteration probe

For abliterated checkpoints, `tools/refusal-probe.py` is the ONLY gate
that fails if the config silently served the stock checkpoint (bench,
GSM8K, RULER all pass either way). It scores refuse-vs-comply on short
greedy generations over a benign set (over-refusal check) and a mild,
category-level sensitive set (non-operational by design - it measures
whether the refusal circuit fires, not content). Runs with `--insecure`
to skip the CA bundle.

## Provenance caveat

Record the exact invocation (concurrency, timeout, limit) alongside any
published number. The original GLM-5.3-Flash 131K RULER figures are in
`docs/GLM53_FLASH.md` but the command that produced them is recorded
nowhere (not git, not session transcripts) - do not assume a prior run's
parameters; measure and write them down.
