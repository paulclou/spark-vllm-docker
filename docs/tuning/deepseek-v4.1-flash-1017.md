# DeepSeek V4.1 Flash 1017 compatibility and qualification

## 2026-10-18: initial recipe configuration

The three- and four-Spark recipes serve
`deepseek-ai/DeepSeek-V4.1-Flash-1017` with the existing `vllm-node-b12x`
image profile and B12X loader, attention, linear, and MoE backends. They
inherit the original recipes' DSpark settings, disk Engram configuration,
five-image request cap, scheduling limits, and memory settings. The three-node
variant retains its `0.5g` KV cache budget. No new performance tuning was run.

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
| Context window | 1M tokens | `--max-model-len auto` retains memory-based sizing, bounded by the model's 1,048,576-token config; full context is not pinned or runtime-qualified. |
| Output budget | At least 256K tokens | Request-driven; the recipe sets no output-token limit. A client must request a suitable budget within the effective context. |

Neither checkpoint supplies `generation_config.json` or sampling overrides
in `config.json`. The [B12X Chat Completions protocol](https://github.com/local-inference-lab/vllm/blob/dev/karmic-kraken/vllm/entrypoints/openai/chat_completion/protocol.py)
uses the sampling fallbacks above when the request omits them. The companion
pi registration uses a 262144-token output budget; omp does not install a
global output clamp and uses the available context when the request omits a
budget. Client requests can override sampling and output size.

These are adaptations of the existing Spark recipes, rather than a claim
that every serving flag is a model-card default. Relative to the matching
non-1017 recipe, the generated vLLM command changes only the checkpoint ID
and adds `--tokenizer-mode deepseek_v41`. Recipe names, descriptions, header
comments, and the four-node short alias identify the new variant; all prior
defaults, environment variables, B12X/DSpark tuning, disk Engram placement,
image cap, and three-node KV budget remain identical.

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
