# DeepSeek V4.1 Flash 1017 compatibility and qualification

## 2026-10-18: initial recipe configuration

The three- and four-Spark recipes serve
`deepseek-ai/DeepSeek-V4.1-Flash-1017` with the existing `vllm-node-b12x`
image profile and B12X loader, attention, linear, and MoE backends. They
inherit the original recipes' DSpark settings, disk Engram configuration,
five-image request cap, scheduling limits, and GPU utilization setting. Both
variants pin `--max-model-len 1048576` to match the official context window.
The three-node variant removes its original `0.5g` KV cap for the source-based
capacity reason below. No new performance tuning was run.

The [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
describes a weight update and ships no Jinja chat template. Both recipes
therefore explicitly pass `--tokenizer-mode deepseek_v41`, alongside the
matching reasoning and tool-call parsers. This mode is accepted by the
[B12X source profile](https://github.com/local-inference-lab/vllm/blob/dev/karmic-kraken/vllm/config/model.py).
The card recommends temperature 1.0 and top_p 1.0; configure sampling in the
client as appropriate for its task.

### Model-card defaults audit

| Setting | Model card | Recipe behavior |
| --- | --- | --- |
| Temperature / top_p | 1.0 / 1.0 | No override; the B12X Chat Completions fallbacks are 1.0 / 1.0. |
| Context window | 1M tokens | Explicit `--max-model-len 1048576`, matching the model config; capacity is not runtime-qualified. |
| Output budget | At least 256K tokens | Request-driven; the recipe sets no output-token limit. A client must request a suitable budget within the effective context. |

Neither checkpoint supplies `generation_config.json` or sampling overrides
in `config.json`. The [B12X Chat Completions protocol](https://github.com/local-inference-lab/vllm/blob/dev/karmic-kraken/vllm/entrypoints/openai/chat_completion/protocol.py)
uses the sampling fallbacks above when the request omits them. The companion
pi registration uses a 262144-token output budget; omp does not install a
global output clamp and uses the available context when the request omits a
budget. Client requests can override sampling and output size.

These are adaptations of the existing Spark recipes, rather than a claim
that every serving flag is a model-card default. Relative to the matching
non-1017 recipe, the generated vLLM command changes the checkpoint ID, adds
`--tokenizer-mode deepseek_v41`, and pins `--max-model-len 1048576` instead of
`auto`. Recipe names, descriptions, header comments, and the four-node short
alias identify the new variant. The three-node variant additionally removes
`--kv-cache-memory-bytes 0.5g` and its recipe default, allowing automatic KV
memory sizing from the unchanged `--gpu-memory-utilization 0.87`. All other
defaults, environment variables, B12X/DSpark tuning, disk Engram placement,
and image cap remain identical.

Metadata inspected at checkpoint revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` matches the original checkpoint's
architecture configuration, tokenizer files, tensor index, and total shard
size. This supports retaining the existing launch settings; it does not
establish runtime compatibility with a particular local B12X image.

Download the complete new model snapshot on every selected node, including
its Engram weights. The final two safetensors shards contain updated Engram
data; reusing old checkpoint files or previously extracted disk tables would
mix checkpoints. If an existing disk-table cache is reused by the installed
loader, verify that it was generated from the new snapshot before serving.
Keep the original snapshot and its artifacts available for rollback.

## 2026-10-18: explicit 1M context and three-node cache limit

The original recipes' automatic context sizing is replaced by an explicit
1,048,576-token limit. This prevents silently selecting a smaller context;
startup can reject the requested size when the available cache is insufficient.

Source inspection of B12X revision
`8244d1c44348ae8a5bbfc9f1b607021652283ed4` predicts that the inherited
three-node `0.5g` cache budget cannot support this explicit limit. The model's
[official config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json)
sets KV source layer 20 to compression ratio 1, and the
[SM120 attention implementation](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/vllm/models/deepseek_v41/attention.py#L464-L466)
stores a 584-byte packed KV record per token with one KV head per rank.
That one source alone requires 584 MiB at 1,048,576 tokens, before padding,
other source layers, index caches, and sliding-window state. Tensor parallel
size three does not divide this single-head cache; these recipes do not enable
decode context parallelism. The
[cache capacity check](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/vllm/v1/core/kv_cache_utils.py)
raises an error when the requested maximum length needs more cache than the
available budget.

The new three-node recipe therefore removes that cap and uses automatic KV
memory sizing from its existing GPU utilization setting, matching the
four-node recipe. Context remains explicitly 1M. The original non-1017
three-node recipe is unchanged.

This is a source-based prediction, not a test of an installed image. Automatic
KV memory sizing does not prove that weights, runtime overhead, and a full
1M-context request fit on three or four Sparks. Both topologies still require
operational memory qualification; startup should refuse an insufficient
budget instead of reducing the requested context.

## Validation boundary

Repository checks and explicit three- and four-node dry-runs validate recipe
parsing and generated commands only. No image was built or pulled, no model
was downloaded, and no container was launched for this change.

Before switching a client, qualify the installed image and complete new
snapshot on the intended topology: confirm startup and `/v1/models` report
the new model ID, then check text, reasoning, streamed tool calls, images
(including two- and five-image history), prefix caching, and memory use.
Record the image digest, model revision, topology, effective context, and
results here when that operational work is requested. Existing measurements
for the original checkpoint do not qualify 1017 performance or accuracy.
