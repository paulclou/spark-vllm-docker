# GLM-5.3-Flash on this cluster

Background for `recipes/glm-5.3-flash-nvfp4.yaml` (the production serve
config) and the reference ports in `docs/reference-recipes/`.

## Model and checkpoint choice

320B total / 18B active multimodal MoE: 34 KDA linear layers + 11 sparse
NoPE-MLA layers + an MTP head; native context 262,144 (1M capable). Dtype fit
on 121 GiB nodes at TP=4:

| Checkpoint | Total | Per node | Verdict |
| --- | --- | --- | --- |
| bf16 original | ~640 GB | ~160 GiB | impossible |
| official FP8 | ~300 GiB | ~75 GiB | unified-memory knife-edge - avoid |
| LibertAIDAI NVFP4 | 182 GiB | ~46 GiB | the recipe until 2026-09-11 (all measurements) |
| nvidia NVFP4 (official, ModelOpt 0.47) | 204 GB | ~51 GiB | the recipe since 2026-09-11 (not yet booted) |

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
   error. `--moe-backend marlin` is mandatory. Root cause per the
   LibertAI card ("check your output before trusting it"): the
   checkpoint originally shipped no MoE input_scale, and vLLM's fused
   NVFP4 path silently defaults it to 1.0; marlin dequantizes weights
   and never reads an activation scale. This also plausibly explains
   tonyd2wild's "intermittent corrupted token IDs" report as a
   serving-stack bug rather than checkpoint damage - which downgrades
   the parked LibertAI-corruption risk for marlin-backed deployments
   like ours.
2. **Autotune false hang**: a 60+ minute "No available shared memory
   broadcast block found" loop at first boot is FlashInfer autotuning, not a
   hang (CPU >150% = still working). Caches persist to `~/.cache/vllm`;
   later boots are fast.

## The mod (removed)

`mods/fix-glm53-nope-rope-pad` adapted GLM's NoPE-MLA to the fp8_ds_mla KV
layout (rope=64 build, q/k_pe zero-pad, kpool top-k table 2176 -> 2048).
It served the original kingjones fp8_ds_mla port; no current recipe uses
that KV layout outside Mia's EXL3 overlay image, which carries the NoPE
handling internally - so the mod was dropped from the branch (recoverable
from git history if an fp8_ds_mla config returns).

## The NVFP4 global-scale mod (mods/fix-nvfp4-moe-global-scale, 2026-09-07)

Both GLM NVFP4 recipes carry this mod. It fixes vLLM
[#54150](https://github.com/vllm-project/vllm/issues/54150) inside the
container: the fused `[w1; w3]` NvFp4 MoE repack takes only w1's per-expert
global scale and merely logs when w3's differs. Our boot log has that line
(`w1_weight_global_scale must match w3_weight_global_scale. Accuracy may be
affected.`), and the checkpoints have the mismatch: measured on the
safetensors, 68.5% of the orcarouter checkpoint's 12,096 gate/up expert
pairs differ (ratio median 1.078, p90 1.23, max 9.96); LibertAIDAI stock is
the same quant (30.9%, max 10.0); RedHatAI is 100% equal by construction
(llm-compressor fuses gate/up scales) and is unaffected. Every w3 weight in a
mismatched expert is dequantized with the wrong global scale, and the model
emits invalid UTF-8 byte-token sequences: U+FFFD in Korean and emoji text at
temperature 0, present on the prefill path (`prompt_logprobs`), so spec
decode, parsers, and tokenizer are exonerated. Found 2026-09-06 by scanning
BFCL responses (16 of 3,641; see the BFCL section once PR #31 lands).

The mod requantizes instead of dropping w3's scale: per expert it shares one
global scale (the per-expert min for compressed-tensors' divisor
`weight_global_scale`, the max for ModelOpt's multiplier `weight_scale_2`)
and folds each shard's ratio, <= 1 by construction, into that shard's E4M3
block scales. The dequantized weight is unchanged up to one extra E4M3
rounding on the moved shard; nothing is clamped. This mirrors vLLM's own FP8
MoE path (`process_fp8_weight_tensor_strategy_moe`, "use the max to
requantize") and matches the two validated variants in the issue thread
(8/6 -> 0/6 U+FFFD on 2x GB10 at our vLLM commit `0.1.dev20051+g487ecf187`).
Fail-closed on anchor drift; test: `tests/test_fix_nvfp4_moe_global_scale_mod.sh`.

Verified live 2026-09-07 (first boot with the mod, 09:16 CDT; the mod
reported `patched` for both loader files on all four nodes and the boot log
line became `w1_weight_global_scale != w3_weight_global_scale; requantizing
the block scales onto a shared per-expert global scale (vLLM #54150)`):

| Probe (temperature 0) | before the mod | with the mod |
| --- | --- | --- |
| Korean ThinQ prompt, chat+tools, U+FFFD per run | 5 / 4 / 5 | 0 / 0 / 0 |
| same prompt, raw `/v1/completions` | 7 / 6 | 0 / 0 |
| forced ` 드릴`: rank / logprob of byte token 250 | rank 3 / -4.03 | rank 1 / -0.00 |
| 18 flagged BFCL ids, responses with U+FFFD (3 passes) | 11 / 9 / 7 | 0 / 0 / 0 |
| 18 flagged BFCL ids, total U+FFFD chars (3 passes) | 65 | 0 |
| raw `<tool_call>` markup in content | 1 of 54 | 0 of 54 |
| DFlash2 accepted tokens per draft (metrics since boot) | 3.3-4.1 | 3.4 |

Rerun recipe: `/tmp/bfcl-harness/repro_ids.json` on the head with
`BFCL_PROJECT_ROOT=<fresh dir> bfcl generate --run-ids`. KV pool at this
boot: 6,679,972 tokens (6.37x at 1M).

Quality gates re-run on the patched build the same day (same protocol as
2026-08-31, head-node localhost over plain HTTP, results under
`~/quality-gates-20260907/` on the head):

| Gate | 2026-08-31 unpatched | 2026-09-07 patched |
| --- | --- | --- |
| GSM8K 200q 5-shot (flex / strict) | 91.0 / 91.0 % (+/-2.0) | 94.5 / 94.5 % (+/-1.6) |
| RULER 8K (s2 / mk1 / vt), 75/75 requests | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| RULER 64K, 75/75 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| RULER 131K, 75/75 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| Refusal benign / sensitive | 0/8 / 0/8 | 0.0% / 0.0% |
| llama-benchy pp2048 / tg128 (3 runs) | 1674 / 56.0 tok/s | 2196 +/- 54 / 54.9 +/- 0.8 tok/s |
| DFlash2 accepted tokens per draft | 3.3-4.1 | 3.4 |

Unchanged within noise on every quality gate (GSM8K +3.5 is inside the
combined error bars). The prefill number is not attributable to the mod,
which runs at weight-load time only: the 08-31 figure was measured through
the TLS launcher over the network, today's on localhost. One false alarm on
the way is worth recording: the first RULER pass came back 0.16-0.32 at 8K
and ~0 at 64K because the documented invocation lacked
`max_length=1050000` and lm-eval silently left-truncated every prompt to
2048 tokens (see the landmine in `docs/BENCHMARKING.md`). Before that was
found, the patched server was cleared directly: identical prompts replayed
by hand scored 10/10 as text and as the same 8,071 token ids, 20/20 at
8-wide concurrency, 10/10 with 92% prefix-cache hits, and the real block
scales (min 5.5, typical 72-352) show zero fp8 underflow after the
requant (effective-scale error 1-2% vs 5-20% unpatched).

## Known limits / tuning levers

- `--language-model-only`: multimodal front-end costs ~15.7 GiB on the API
  node; vision stays off until that headroom is proven at TP=4.
- `max_num_seqs: 1` is the validated single-stream config; raising it is the
  first tuning experiment at TP=4 (watch the MLA + KDA state pools).
- MTP-5 speculative decoding is in the base config (validated ~1.7-2x).

## The MM variant (docs/reference-recipes/glm-5.3-flash-nvfp4-mm.yaml)

Replicates MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark (MIT, @aed98a13ca75):
multimodal on, max_num_seqs 8, fp8_e4m3 KV, MTP-4, Ray executor (launch with
--ray), 23-30 tok/s x1 / 72 aggregate x8 measured. The build context
(Dockerfile.glm53-mm + docker/glm53/, vendored from her repo) was removed
with the never-built variant - recover from MIT upstream @aed98a13ca75 or
git history. It folded Mia's two layers into one build: FlashInfer pinned to 0.6.18 SM90-NoPE-MLA (the stock image's
SM120 sparse path assumes pe_dim=64; this checkpoint is NoPE), NCCL 2.30.7,
cutlass-dsl 4.6.2, her 423-line SM121 source patch, Ray 2.58, and the
model's chat template baked at /opt/glm53/. That baked template is
byte-identical (verified 2026-08-30) to both the LibertAI checkpoint's
chat_template.jinja and zai-org's official - so the --chat-template flag
is defensive redundancy against sparse/broken snapshots, not a
customization; vLLM autoloads the same template from the checkpoint,
which is what the production recipe relies on (no flag). The
verification is also a small counter-datapoint on the LibertAI
corruption risk: the repack's tokenizer-side files are faithful to the
official repo (the claim concerns quantized weights, so it stays
parked).

Her hard-won operational notes: NCCL must be pinned to the CX7 interfaces
with NCCL_IB_GID_INDEX=3 or ncclCommInitRank busy-waits forever;
gpu_memory_utilization 0.84 (0.90 fails the free-memory check on UMA);
--enforce-eager + --moe-backend marlin is the deployed-stable MoE path (her
native path fallback triggers on cudaErrorNoKernelImageForDevice - loud,
unlike the silent FLASHINFER_CUTLASS garbage the text-only variant guards
against).

Superseded for images: the production recipe serves MM natively (validated
2026-08-30). This port remains the only Ray/concurrency-validated MM path
(author-validated at TP2; never booted on our cluster).

## The EXL3 variant (docs/reference-recipes/glm-5.3-flash-exl3.yaml)

Replicates MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (@c91754f151ce): EXL3
kernels executing inside vLLM's serving layer via Mia's prebuilt overlay
image (ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3, FROM the
dedicated glm53 image; includes the aarch64 AVX-stub compile patch, NoPE
zero-pad into fp8_ds_mla geometry, and video-placeholder fixes).

Why it was considered (superseded by the TP=4 measurements below - the
NVFP4 production recipe wins every speed metric): Mia's GB10 numbers:
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

## The serve recipe (recipes/glm-5.3-flash-nvfp4.yaml, 2026-08-30)

(Formerly glm-5.3-flash-serve.yaml; renamed when the reference ports moved
to docs/reference-recipes/ and this became the sole launchable GLM recipe.
The tony-flags bench config it was measured against is now
docs/reference-recipes/glm-5.3-flash-nvfp4-bench.yaml.)

Production config, built ground-up (2026-08-30) as a replication of the
official vLLM recipe (recipes.vllm.ai GLM-5.3-Flash, GB200 NVL4 profile);
on 2026-09-11 the recipe was switched to NVIDIA's own checkpoint and
model-card setup (see "NVIDIA official setup port" below - not yet booted
in that form; every measurement in this section is from the 2026-08-30
LibertAIDAI form). The 2026-08-30 form replicated the vLLM recipe with
exactly three contract exceptions: image (tonyd2wild sm121-v11-dflash2 - official
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
  vLLM build - the LibertAI card documents the same field quirk.
- deepseek_r1 is the card-MANDATED vLLM parser: the card documents that
  stock vLLM's glm45 resolves to the GLM-4.7 parser, expects <think> in
  the output while this template emits it as the last PROMPT token, and
  "silently discards the whole reply" (content and reasoning_content
  both empty). Nuance: on tonyd2wild's custom build glm45 demonstrably
  produced content (our bench campaign ran on it), so his image likely
  patches the parser - but on stock images glm45 is a landmine, and the
  official vLLM recipe + both Mia configs carry it.
- tony's launcher passes enable_thinking:false - a confirmed no-op: the
  checkpoint template handles only reasoning_effort and clear_thinking.
  His published numbers therefore ran with full max-effort thinking.
- The checkpoint's generation_config.json carries temperature 1.0 /
  top_p 0.95 (zai's own eval sampling) - vLLM serves those as the
  sampling defaults for clients that set none. Aligned by default; our
  temp-0 runs were explicit benchmark choices.

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

### Reasoning-effort dial - measured (2026-08-30, live endpoint)

12-cell matrix: low/high/max/unset x three difficulty tiers, greedy
(temperature 0), single stream. Times include API overhead.

| Prompt tier | low | high | max | default (unset) |
| --- | --- | --- | --- | --- |
| easy arithmetic | 16 tok, 0.5s, OK | 20 tok, 0.6s, OK | 55 tok, 1.0s, OK | 55 tok, 1.0s, OK |
| 3-way logic puzzle | 114 tok, 2.4s, OK | 236 tok, 4.3s, OK | 211 tok, 3.6s, OK | 328 tok, 7.0s, OK |
| competition math (AIME-style) | 444 tok, 9.4s, WRONG | 863 tok, 16.9s, WRONG | 7606 tok, 122s, OK | 7997 tok, 124s, OK |

Findings:

- **default == max**, confirmed behaviorally (identical outputs on the
  easy tier; same solve + same-scale budget on the hard tier). The
  template maps anything outside ['low','high'] to max.
- **Effort scales with difficulty, dramatically at the top**: on hard
  problems max thinks ~9-17x longer than high - and was the ONLY level
  that answered correctly (738; low said 657, high said 648, both
  confidently wrong). Token growth low->high->max is monotone on easy
  and hard; on the mid tier high ~= max within run noise.
- **Do NOT default the server to high**: it reads as "max but cheaper"
  until a genuinely hard problem arrives, where it fails where max
  succeeds. Keep the server default at max (the recipe does, by
  omission); latency-sensitive clients opt DOWN per request with
  chat_template_kwargs {"reasoning_effort": "low"|"high"}.
- **Temp-0 is not bit-deterministic on this stack**: a repeated cell
  gave 243 vs 236 tokens, and unset-vs-max diverged on the mid tier
  (328 vs 211) despite identical templates - continuous-batching /
  MoE reduction-order float noise, plus probabilistic drafting.
  Expect ~5-50% token-count variance between identical greedy runs;
  don't read single-run token counts as exact.

MM inference VALIDATED on the serve config (2026-08-30 smoke probes,
tony image, no --language-model-only): a shapes/colors/text image and a
bar chart both described exactly (all shapes, positions, colors, the
rendered word, chart values and the odd-colored bar) in 1.5-2.8 s at
~300-385 prompt tokens. The vision front-end costs no extra flags on
this stack; probe scripts in the session scratchpad.

Still unmeasured on the serve config: >131K prompts, DFlash2 acceptance
under 16-seat concurrency, MM under load/large images (probes were
smoke-grade).

### NVIDIA official setup port (2026-09-11, not yet booted)

NVIDIA published `nvidia/GLM-5.3-Flash-NVFP4` (MIT, 2026-09-09): ModelOpt
v0.47.0, recipe `nvfp4_experts_dense_mlp-kv_fp8_cast` - routed experts plus
the dense MLPs of layers 0-2 in NVFP4, FP8 KV cast; attention, shared
experts, router, embeddings, lm_head and the vision tower stay bf16.
Calibrated on CNN DailyMail + Nemotron-Post-Training-v2. 204 GB on disk
(~51 GiB/node at TP=4 vs ~46 for LibertAIDAI), 33 shards. config.json:
`Glm5NextForConditionalGeneration`, `model_type glm5_next`, no `auto_map`
(no remote code in the repo), `num_nextn_predict_layers: 1` (one MTP layer
shipped), `transformers_version 5.16.1`. Card requires
`transformers>=5.16.1`; the `vllm-node-glm5.3-flash` image carries 5.15.1
- verified NOT to matter (2026-09-11): vLLM ships its own `Glm5NextConfig`
(`transformers_utils/configs/glm5_next.py`) and `get_config()` on the
staged NVIDIA snapshot returns it with `quant_method modelopt`, with and
without `trust_remote_code`, in both this image and the official arm64
image. Plain `transformers.AutoConfig` does fail (`glm5_next` unknown to
5.15.1), which is what the card's pin is about. Card
accuracy vs bf16 (temp 1.0, top_p 0.95): GPQA-D 0.9211/0.9217, SciCode
0.5769/0.5621, MMMU-Pro 0.763/0.7688, AA-LCR 0.7106/0.71, IFBench
0.6054/0.613, Terminal Bench 2.1 0.8315/0.8258. Tested hardware: one GB200
node, TP=4, DP=1.

The serve recipe now uses this checkpoint (owner decision, 2026-09-11) and
NVIDIA's serving setup, matching the card's command wherever GB10 allows
(owner direction: prefer the official recipe, then validate with the
benchmark suite). Weights are NOT in any node's HF cache yet (the JSON
files are staged on the head, snapshot `09b04e5e`); a per-node download
precedes the first boot.

On the image: `vllm-node-glm5.3-flash` (tonyd2wild sm121-v11-dflash2) is
the official `vllm/vllm-openai:glm53-flash-arm64-cu130` image (also present
on the head) plus a patch layer, verified from `docker history` and
`pip list`: same vLLM `0.1.dev20051+g487ecf187`, torch 2.13.0+cu130,
transformers 5.15.1, xgrammar 0.2.3, CUDA arch list sm_80..sm_120; the
delta is flashinfer 0.6.18.dev20260819 (official: 0.6.17), the DFlash2
speculator package, `qwen3_dflash2.py`, and patches
`patch_registry_and_select`, `patch_glm_aux_capture`, `patch_kv_page_lcm2`,
`patch_glm5_drafter_group`, `patch_v7`, `patch_v8_fp8`. So the image is
not a fork of a different vLLM; it is the official arm64 build with DFlash2
bolted on. NVIDIA's x86 GB200 path is the only thing that cannot run here
(aarch64). Both
checkpoints are ModelOpt quants (LibertAIDAI v0.45, NVIDIA v0.47) with
per-expert `experts.N.{gate,up,down}_proj.weight` + `weight_scale` +
`weight_scale_2` tensors, so the loader path, the EP weight filter, and
the global-scale mod all carry over unchanged; the mod's 68.5% mismatch
figure was measured on LibertAIDAI/orcarouter and is unmeasured on
NVIDIA's. NVIDIA additionally quantizes the dense MLPs of layers 0-2. The
DFlash2 drafter was trained against the bf16 base model, so it is
target-compatible; acceptance may shift from the LibertAIDAI figure. The
card's vLLM command, verbatim:

```
pip install -U "transformers>=5.16.1" && \
vllm serve /checkpoint --served-model-name nvidia/GLM-5.3-Flash-NVFP4 \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --data-parallel-size 1 \
  --enable-expert-parallel --enable-ep-weight-filter --reasoning-parser glm45 \
  --kv-cache-dtype fp8 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --max-num-seqs 32 \
  --gpu-memory-utilization 0.90
```

The card's SGLang command additionally names `--tool-call-parser glm47` and
`--mem-fraction-static 0.85`, which the recipe already matches.

Flag-by-flag disposition (image `vllm-node-glm5.3-flash`, vLLM
`0.1.dev20051+g487ecf187`, inspected in the container 2026-09-11):

| NVIDIA flag | Recipe | Why |
| --- | --- | --- |
| `--tensor-parallel-size 4` | same | 4 x 1-GPU Sparks instead of 4 GPUs in one GB200; wiring by launch-cluster.sh |
| `--data-parallel-size 1` | stated (`data_parallel`) | matches the card |
| `--enable-expert-parallel` | adopted | New here. Experts sharded across the 4 ranks instead of column-split. `modelopt.py` `ModelOptNvFp4FusedMoE` passes `layer.expert_map` to the kernel and `experts/marlin_moe.py` (our forced `--moe-backend marlin`) honors `expert_map`/`global_num_experts`. With DP=1 no all2all backend engages (`ParallelConfig.use_all2all` is False), so MoE traffic stays on the existing NCCL TP group. Correctness/perf on GB10 unmeasured. |
| `--enable-ep-weight-filter` | adopted | `default_loader.py` `_init_ep_weight_filter` skips non-local expert tensors at read time; LibertAIDAI's per-expert layout qualifies. Should cut per-node load I/O ~4x. Unmeasured. |
| `--reasoning-parser glm45` | adopted (was deepseek_r1) | In this image `glm45` and `glm47` both resolve to `Glm47MoeParserReasoningAdapter` (parser-engine adapter, not the legacy parser the LibertAI card calls a landmine). The 2026-08-28 bench campaign ran glm45 on this image with content present. Re-verify `message.reasoning` on first boot. |
| `--kv-cache-dtype fp8` | same | |
| `--model-loader-extra-config enable_multithread_load` | adopted, `num_threads` 128 | Present in `default_loader.py`. Matched to the card; threads are I/O-bound, so 128 on 20 cores is over-subscription, not a fault. Recipe default `loader_threads`; watch host memory during load on first boot (UMA). |
| `--max-num-batched-tokens 8192` | same | |
| `--enable-chunked-prefill` | stated | vLLM default; stated like NVIDIA does |
| `--max-num-seqs 32` | 32 | matched to the card (2026-09-11); the 2026-08-30 fleet cap of 16 is superseded for this recipe |
| `--gpu-memory-utilization 0.90` | 0.85 | GB10 boot-mandatory (UMA headroom), see above |

Dropped to match the card (2026-09-11): `--trust-remote-code` (NVIDIA's
repo has no `auto_map`; vLLM's own config class loads it, verified above)
and `--block-size 2304`. On block size: the 2026-08-30 note claimed the
default is 16; it is not. `CacheConfig.block_size` defaults to None and
resolves to the attention backend's kernel block size - the SM120 sparse
MLA backend (`FLASHINFER_MLA_SPARSE_SM120`) supports [64, 256] - so the
card's command runs on a backend-chosen block. 2304 (9 x 256) was a
measured prefill win (best 72K needle, 36.6s vs 42s) and is the first
thing to A/B back in if the benchmark shows a long-context regression.

Not in NVIDIA's command, retained: `HF_HUB_OFFLINE=1` (our equivalent of
the card's local `/checkpoint` path; boot-mandatory for the processor),
`--moe-backend marlin` (sm_121 auto-select is silent garbage; the card ran
on sm_100 where auto-select picks a working FlashInfer kernel), two GB10
mods (see below), `--max-model-len 1048576`, `--enable-prefix-caching`,
glm47 tool parser + auto tool choice (the card's SGLang section names
glm47; vLLM's `--enable-auto-tool-choice` requires a parser), DFlash2 k=7
(owner choice; NVIDIA's command has no speculation, but the checkpoint
ships its MTP layer - 889 `layers.45` tensors - so `{"method":"mtp"}` is
the commercial fallback; the DFlash2 drafter is dense, so EP does not touch
it), `clear_thinking`, and the `glm-5.3-flash-nvfp4` alias.

Mods, are they still needed on NVIDIA's checkpoint:

- `fix-glm53-topk-sm120`: yes. GB10 kernel-launch limit in the sparse
  attention indexer, independent of checkpoint; >32K prompts crash
  without it. The official arm64 image has the same gap.
- `fix-nvfp4-moe-global-scale`: NOT applied to this recipe. vLLM #54150
  is still open upstream (main's `modelopt.py` still only warns, checked
  2026-09-11), so the mod remains the only fix for affected checkpoints -
  but NVIDIA's is not one. Measured 2026-09-11 by HTTP range-reading
  `weight_scale_2` (F32 scalars) for every expert of layers 5, 20 and 40:
  288/288 gate==up in each layer, ratio max 1.000 (LibertAIDAI/orcarouter:
  31.5% equal, max 9.96). Consistent with the quant summary's single fused
  `gate_up_proj_weight_quantizers.N` per expert, so equality holds by
  construction for every layer, including the MTP layer 45. The mod would
  be an exact no-op; it is left out to keep the recipe at the official
  baseline plus GB10 fixes only. Re-add if the checkpoint changes.
- `mia-backports-20260830`: yes. Both are image-level bugs (kpool tail
  slot map; XGrammar termination under speculative batches) present at
  this vLLM commit in the official image too.

Validation status: dry-run only (`tests/test_recipes.sh` green). Nothing
in this form has booted and the weights are not downloaded. First launch
must be foreground (not the systemd unit) and must re-run the smoke gates
above before switching over: boot log free of EP/loader errors, `message.reasoning` populated under glm45,
glm47 tool call structured, 72K needle, the U+FFFD scan from the
global-scale mod section, and llama-benchy pp2048/tg128 against the
serve-config table (EP changes the MoE communication pattern, so decode may
move either way). If EP misbehaves, drop the two EP flags first; if glm45
discards replies, revert to deepseek_r1 - both single-line changes, neither
undoes the rest of the port. The uncensored recipe is deliberately NOT
ported (owner decision, 2026-09-11) and stays on the 2026-08-31 form.

## The Uncensored variant (recipes/glm-5.3-flash-uncensored-nvfp4.yaml, 2026-08-31)

Byte-identical to the serve recipe except the checkpoint:
orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 (MIT), an experts-only NVFP4
re-quant of OrcaRouter's abliterated GLM-5.3-Flash. Card-reported quality
vs the official FP8 reference: PPL +3.8%, KLD 0.073, top-1 agreement
91.7%; harmful-refusal rate ~17.2% (vs ~90% stock), benign over-refusal
0%. Because the config is unchanged, every measurement below is a delta
against the base-recipe table above.

### Model-card deviations (deliberate)

The card's launch example (official x86 image, 8xH100 TP8) sets
`VLLM_SSM_CONV_STATE_LAYOUT=DS` and `VLLM_KV_CACHE_LAYOUT=HND`. We set
neither, verified against this build's source and boot log:

- `VLLM_KV_CACHE_LAYOUT=HND` is a no-op here: the selector already picks
  HND for the FLASHINFER_MLA_SPARSE backend (boot log states it).
- `VLLM_SSM_CONV_STATE_LAYOUT` defaults to SD; both layouts are correct.
  The only hard `DS` requirement in vLLM is the NIXL KV-transfer
  connector (disaggregated prefill), which we do not run.
- Empirical backstop: the base recipe ran this stack without either var
  and scored perfect RULER retrieval through 131K.

The card's "Hopper does not support FP8 KV" warning does not apply to
GB10 (sm_121); fp8 KV was validated on the base recipe.

### Measured results - 4x Spark TP=4 (2026-08-31)

Same protocol as the base campaign. Boot: full 1M window served
(assert-window clean), KV pool 6,596,420 tokens (6.29x concurrency at
1M), reasoning field and glm47 tool calls verified.

| Metric | Uncensored NVFP4 | Base NVFP4 (2026-08-30) |
| --- | --- | --- |
| Prefill pp2048 | 1674 +/- 74 tok/s | 1792 +/- 89 tok/s |
| Decode tg128 | 56.0 +/- 7.3 (peak 68) | 52.5 +/- 0.5 (peak 66) |
| TTFT @2K | 1.23 s | 1.15 s |
| DFlash2 acceptance (pos-0 / tok-per-step) | 0.751 / 3.24 (temp 0.7) | 0.918 / 6.43 (Mia EXL3 figure) |
| GSM8K 200q (flex/strict) | 91.0 / 91.0 % (+/-2.0) | 89.0 / 87.5 % |
| RULER 8K (s2/mk1/vt) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| RULER 64K (>32K, topk mod) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| RULER 131K | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 @131072 |
| Refusal - benign over-refusal | 0/8 (0%) | not measured (stock model) |
| Refusal - sensitive tier | 0/8 (0%) | ~90% on stock GLM |
| MM vision smoke (shapes+text, bar chart) | described exactly | validated (base recipe) |

The acceptance drop (0.751 vs the ~0.918 stock reference) is the expected
consequence of drafting with the original-weights DFlash2 against an
abliterated target - the card's 91.7% top-1 agreement predicts it. It is
not a correctness concern: standard rejection sampling keeps outputs
lossless regardless of acceptance. The two acceptance figures are not a
clean A/B (ours: temp 0.7, few prompts; the 0.918 is the Mia EXL3 number
under different conditions), and single-stream decode variance here is
high (+/-7.3), so we do not claim a precise decode delta - measured
decode sits within noise of the base recipe.

k stays at 7 (not swept down): DFlash2 drafts all speculative tokens in
ONE parallel forward pass (vllm .../spec_decode/dflash/speculator.py:
"DFlash processes all speculative tokens in one forward pass"), so the
drafter cost is ~flat in k. Lowering k would only discard accepted tokens
(positions 4-6 still contribute ~0.42 tok/step) for no compute saving, so
k=7 is at or near optimal for a parallel-draft drafter; no-spec is the
slow floor. A downward k-sweep was therefore judged not worthwhile.

### Repeatable measurement protocol

The harness mechanics - TLS via the tailnet URL, llama-benchy and lm-eval
invocations, the lm-eval landmines (validate request count; one RULER
length per invocation; long-context `num_concurrent<=2` + `timeout=3600`;
never pipe through `head`), and reading spec-decode acceptance off
/metrics - are in `docs/BENCHMARKING.md`. GLM-specific parameters:

- served-model-name `glm-5.3-flash-uncensored-nvfp4`; snapshot under
  `models--orcarouter--GLM-5.3-Flash-Uncensored-NVFP4`.
- Speed: llama-benchy `--pp 2048 --tg 128 --runs 3`.
- GSM8K: 200q, `--num_fewshot 5` (short-context, num_concurrent=8 fine).
- RULER: `niah_single_2,niah_multikey_1,ruler_vt`, `--gen_kwargs
  max_gen_toks=256`, `--limit 25`, one length per run at 8192 / 65536 /
  131072. 131072 requires `num_concurrent<=2 timeout=3600` (see
  BENCHMARKING.md) - it will fail with "Session is closed" otherwise.
- Refusal: `tools/refusal-probe.py --insecure` - measurement-grade
  tripwire for the abliteration (the only gate that fails if the config
  silently served the stock checkpoint). Uncensored recipe: 0/8 benign,
  0/8 sensitive (mild tier), consistent with the card's residual ~17% on
  a harder set.
- MM vision: the checkpoint keeps the bf16 visual tower, so a PIL-drawn
  shapes/text image and a bar chart sent as base64 data URIs to
  /v1/chat/completions are described exactly (no extra serve flags needed;
  matches the base recipe's MM validation).
- Tool calling: `tools/bfcl-bench.sh glm-5.3-flash-uncensored-nvfp4
  single_turn` on the head node (see below).

### Tool calling - BFCL v4 single_turn (2026-09-06, live endpoint)

Run on the head against 127.0.0.1:8000 on the live production config (no
restarts; DFlash2 k=7, fp8 KV, glm47 parser, reasoning on). Invocation:
`THREADS=32 tools/bfcl-bench.sh glm-5.3-flash-uncensored-nvfp4 single_turn`
with `bfcl-eval==2026.3.23`. 3,401 cases, 1 request timeout
(live_irrelevance_342-81-3, scored as wrong). Results stay on the head
under `/tmp/bfcl-harness/result-glm-5.3-flash-uncensored-nvfp4/score/`.

| BFCL v4 category | Accuracy | n |
| --- | --- | --- |
| Non-live overall (AST) | 88.00% | 1,390 |
| - simple: Python / Java / JavaScript | 96.0 / 61.0 / 74.0% | 400 / 100 / 50 |
| - multiple / parallel / parallel-multiple | 95.5 / 91.0 / 88.5% | 200 each |
| - irrelevance detection | 60.42% | 240 |
| Live overall (AST) | 79.64% | 1,127 |
| - simple / multiple / parallel / parallel-multiple | 87.98 / 77.97 / 75.0 / 66.67% | 258 / 1,053 / 16 / 24 |
| - irrelevance / relevance detection | 67.65 / 87.50% | 884 / 16 |
| Latency mean / p95 (reasoning on) | 35.9 s / 86.0 s | - |

Ignore the CSV's "Overall Acc 23.17%": BFCL averages in the multi_turn and
agentic categories, which were not run, as zero. The misses are wrong
arguments and, above all, calling a tool when none applied (irrelevance
60-68%) - model behaviour, not a serving defect. Prompts are all under
~4K tokens, so this is a short-context baseline for the tool-call path
only; it says nothing about the >150K garble by itself. A scan of the raw
responses did, however, surface two short-context defects (next section).

### Short-context garble found in the BFCL responses (2026-09-06)

Scanning all 3,641 stored BFCL responses for garble signatures (CJK ratio,
repetition, U+FFFD, leaked markup) found two real, reproducible effects on
the live production server. Neither is length-related.

**1. Dropped UTF-8 bytes -> U+FFFD (16 of 3,641 responses, 0.44%).** Every
hit is in Korean text or at an emoji, in categories where the live
dataset has Korean prompts. Anatomy, verified with `logprobs` +
`return_token_ids` at temperature 0: the tokenizer encodes rare syllables
as byte-fallback tokens (e.g. ` 드릴` = ids 55463 `' \xeb\x93'`, 250
`'\x9c'`, 20058 `'\xeb\xa6'`, 112 `'\xb4'`). The model emits 55463 and
then jumps straight to 20058 with p~0.98; byte token 250 is not in its
top-5. Same at ` 냉방` (drops id 231) and at emoji (drops the last byte).
The detokenizer then renders U+FFFD, e.g. `안내해 �릴게요`.
- Every emitted token is the target's own top-1 (0 non-argmax positions in
  300 and 388 token responses), and the same defect appears in
  `prompt_logprobs` when the canonical text is forced into the prompt
  (id 250 rank 3, lp -4.03, vs the skipping continuation at -0.28). So
  speculative decoding, decode kernels, and CUDA graphs are exonerated:
  the served model's distribution is wrong on the prefill path too.
- Raw `/v1/completions` shows the same U+FFFD counts, so the glm47 parser
  and reasoning parser are exonerated.
- The tokenizer is byte-identical (sha256 19e77364...) across the
  uncensored, LibertAIDAI stock NVFP4, and Mia EXL3 snapshots, so it is
  not a tokenizer mismatch.
- Not deterministic run to run at temperature 0 (3 reruns of the 18
  affected cases: 11/9/7 still broken, positions move) - the server is
  non-deterministic at greedy even when idle, so near-ties flip.
- **Root cause: vLLM #54150, the fused-MoE NVFP4 single-global-scale
  bug** - and PR #30's rule-out of it was wrong. `compressed_tensors_moe_w4a4_nvfp4.py`
  `process_weights_after_loading` repacks the fused `[gate; up]` expert
  GEMM with ONE global scale, gate's (`w13_weight_global_scale[:, 0]`),
  and merely logs when up's differs. Our boot log (2026-09-03 06:42:33)
  carries that log line: `w1_weight_global_scale must match
  w3_weight_global_scale. Accuracy may be affected.` Measured over the
  orcarouter checkpoint's safetensors: 12,096 gate/up expert pairs, only
  31.5% equal; up/gate ratio mean 1.095, median 1.078, p90 1.23, max
  9.96 - the same distribution mechramc measured on a ModelOpt checkpoint
  in the issue (30.9% equal, max 10.0). So orcarouter is a ModelOpt-style
  per-tensor-amax quant re-exported as compressed-tensors; the format
  changed the loader path but not the bug, which lives in both loaders.
  Every expert's up projection is mis-scaled by up to 10x, which is the
  logit damage that drops bytes. The issue's reporter and two
  independent reproductions (one on 2x GB10, same vLLM commit
  `0.1.dev20051+g487ecf187` as ours) show 0 U+FFFD once the scales are
  reconciled; RedHatAI's llm-compressor checkpoint is immune only because
  its gate/up scales are equal by construction. LibertAIDAI stock (our
  base recipe's checkpoint, ModelOpt) is affected too.
- Fix options, both need a restart: (a) a `mods/` patch to
  `compressed_tensors_moe_w4a4_nvfp4.py` that requantizes the up half's
  E4M3 block scales onto a shared per-expert global scale (two validated
  variants are in the issue thread; ~20 lines; no upstream PR as of
  2026-09-06, issue open); (b) a checkpoint whose gate/up scales are
  equal (RedHatAI stock; no uncensored equivalent known).
- Relevance to the agent garble: this is a constant, short-context source
  of exactly the "stray CJK/emoji, 1-3 chars" tail noise seen in the
  poisoned OMP session, and tonyd2wild's recipe notes describe the same
  bug's agent-side face: "when a corrupted token lands inside a tool-call
  block the parser desyncs and generation can spiral into a repetition
  lock." Once such noise is in the transcript the in-context-imitation
  lock-in takes over.

**2. Misnamed tool -> raw markup returned as content (2 of 3,401).**
In `parallel_173` and `parallel_multiple_95` the model wrote
`investment_predictProFit` and `cosine_similarity_calculator` for the
declared `investment_predictProfit` / `cosine_similarity_calculate`. The
glm47 parser runs with `validate_tool_names=True`, so the frame is
rejected and the entire output, `<tool_call>...</tool_call>` markup
included, comes back as `content`. Reproduced 1/3 reruns (with a
different misspelling, `cosine_similarity_cal`). For an agent client
this is the "tool-call markup leaked as text" event that poisons a
session; the fix is client-side healing or a parser that surfaces the
rejected frame instead of dumping it into content.

Repro tooling stays on the head: `/tmp/bfcl-harness/repro_ids.json`
(the 18 ids) with `BFCL_PROJECT_ROOT=<dir> bfcl generate --run-ids`, and
`/tmp/entry_149.json` (the Korean ThinQ prompt) for direct probes.

Run mechanics learned: the server caps running requests at 16, so
`THREADS` above 16 only queues (32 was used; 16 ran). At 16 streams the
full single_turn set took ~60 min plus a ~1 min evaluate. BFCL resumes:
existing result files are loaded and their ids skipped, so a killed run
loses only in-flight requests.

## Garble investigation - single-turn serving exonerated (2026-09-06)

Symptom: the uncensored endpoint episodically emits gibberish in agent
sessions, perceived onset "random after ~150K+ context", multiple evenings
(latest 2026-09-05). Investigated 2026-09-06; the result is a negative that
narrows the search, not a fix.

Server-side forensics for the 2026-09-05 evening: healthy on every known
signature. One unit and one engine per node (no sibling-unit crash loop, no
leaked engines), 0 restarts, zero journal warnings, DFlash2 mean acceptance
length 3.3-4.1 all night with no sustained collapse, KV usage <=8.7%. The KV
pool is 6,548,378 tokens (boot log), so the observed 2.0-4.3% single-request
usage corresponds to real 130-275K-token requests - the reported regime was
genuinely exercised while metrics stayed clean.

Ruled out the same day:
- Stale/divergent image: local `vllm-node-glm5.3-flash` is byte-identical
  (RepoDigest sha256:4def0ef6...) to tonyd2wild's newest published tag
  (`sm121-v11-dflash2`; only v8 exists besides it).
- ~~ModelOpt NVFP4 token corruption (vLLM #54150): our checkpoint's
  `quant_method` is `compressed-tensors`, not ModelOpt.~~ **Wrong, retracted
  2026-09-07.** The format is not the discriminator; equality of the
  gate/up (w1/w3) per-expert global scales is, and orcarouter's differ on
  68.5% of experts (max 9.96x) exactly like LibertAIDAI's. The
  compressed-tensors loader has the same single-gscale shortcut as the
  ModelOpt one, and our boot log carried its warning. This is a real,
  constant, short-context garble source (U+FFFD in Korean/emoji), found by
  the BFCL response scan and fixed by `mods/fix-nvfp4-moe-global-scale`;
  see "The NVFP4 global-scale mod" and the BFCL sections.
- tonyd2wild changelog: no garble reports on the GLM/DFlash2 lane at all;
  his topkfix/CUDA-graph finding (deadlock, not garble) is a different mode.

Reproduction probe (30/30 clean). Deterministic filler document with three
embedded passphrases; task = exact retrieval + per-section summaries;
automated garble detectors (needle miss, >=25x repeated 4-gram, >2% CJK
ratio, special-token leak) over reasoning + answer; unique per-run salt
defeats prefix-cache reuse. Run on the head against 127.0.0.1:8000 on the
live production server (no restarts, deployed config incl. CUDA graphs
FULL_AND_PIECEWISE, fp8 KV, DFlash2 k=7):

| Phase | Variables | Result |
| --- | --- | --- |
| 1 | 120K/160K/200K x5, temp 0, concurrency 2 | 15/15 clean |
| 1b | 160K/200K x5, temp 1.0 top_p 0.95 | 10/10 clean |
| 1c | 160K x5, temp 1.0 + continuous short-request churn (batch condense under a decoding long request) | 5/5 clean |

Probe landmines (for reruns): vLLM's `/tokenize` is at the server root, not
under `/v1`; an empty `content` with `finish_reason=length` means the
reasoning phase exhausted `max_tokens` (raise to 3072 and read
`reasoning_content`) - it is not garble; a deliberately repetitive filler
makes the model legitimately repeat a template sentence, so a repetition
threshold of 10 false-positives (use >=25). Tooling lives on the head node:
`~/probe_glm_garble.py`, outputs in `~/probe-glm-garble/` (not vendored).

Conclusion: context length (to 200K), production sampling, and ragged
batching are each exonerated for single-turn serving. What production has
and the probe does not is the multi-turn agentic path: streamed responses
with accumulated tool-call markup and thinking blocks. Leading hypothesis is
the client-side markup-leak lock-in class (one malformed tool-call frame the
`glm47` parser misses lands as text, is replayed every turn, and degrades
the session) - which also reframes the "150K+" correlation as turn count,
not KV depth: more turns, more chances for one bad frame. Next step requires
a captured garbled transcript (client + timestamp): raw tool-call markup in
it implicates the client healer path; token salad mid-thinking with no
markup implicates the server, and only then is the config knob matrix
(enforce-eager / no-spec / fusion passes off) worth its restarts.

Postscript (2026-09-07): the "single-turn serving exonerated" verdict held
only for what this probe measured - English filler, no tool calls, detectors
tuned for salad and repetition. A scan of 3,641 BFCL responses the next day
found a constant short-context defect the probe could not see: dropped
UTF-8 bytes in Korean/emoji text (vLLM #54150, the item retracted above),
plus misnamed tool calls whose whole frame is returned as text by the glm47
parser's name validation. Both are exactly the "stray CJK/emoji, 1-3 chars"
and "markup leaked as text" ingredients of the poisoned OMP session. The
byte-drop is fixed (mod verified live 2026-09-07); whether the agent garble
disappears with it is the next thing to watch.
