# Recipes

Recipes provide a **one-click solution** for deploying models with pre-configured settings. Each recipe is a YAML file that specifies:

- HuggingFace model to download
- Container image and build arguments
- Required mods/patches
- Default parameters (port, host, tensor parallelism, etc.)
- Environment variables
- The vLLM serve command

## Quick Start

```bash
# List available recipes
./run-recipe.sh --list

# Run a recipe in solo mode (single node)
./run-recipe.sh glm-4.7-flash-awq --solo

# Full setup: build container + download model + run
./run-recipe.sh glm-4.7-flash-awq --solo --setup

# Run with overrides
./run-recipe.sh glm-4.7-flash-awq --solo --port 9000 --gpu-mem 0.8

# First cluster deployment: discover once, then run from saved configuration
./run-recipe.sh --discover
./run-recipe.sh minimax-m2-awq --setup
```

## Cluster Node Discovery

Autodiscovery is the default way to configure cluster nodes:

```bash
# Auto-discover nodes and save to .env
./run-recipe.sh --discover

# Run recipe (uses nodes from .env automatically)
./run-recipe.sh minimax-m2-awq --setup
```

When you run `--discover`, it:
1. Detects active CX7 interfaces and determines mesh vs. standard topology.
2. Scans the network for peers that are both SSH-reachable **and** have an NVIDIA GB10 GPU.
3. In mesh mode, separately discovers `COPY_HOSTS` on the direct IB-attached interfaces.
4. Prompts for per-node confirmation for `CLUSTER_NODES` and `COPY_HOSTS`.
5. Saves the full configuration (including mesh NCCL settings if applicable) to `.env`.

Future recipe runs automatically use nodes from `.env`. Do not pass IP addresses
with `-n` unless you were specifically instructed to use a manual node list or
the network topology prevents autodiscovery from working.

When distributing the container image or model files, the runner uses `COPY_HOSTS` from `.env` (which may differ from `CLUSTER_NODES` in mesh mode) to ensure transfers go over the fastest available path.

## Workflow Modes

### Solo Mode (Single Node)
```bash
# Explicitly run in solo mode
./run-recipe.sh glm-4.7-flash-awq --solo
```

### Cluster Mode (Multiple Nodes)
```bash
# Discover nodes and network topology on first use
./run-recipe.sh --discover  # First time only

# Deploy using the saved .env automatically
./run-recipe.sh minimax-m2-awq --setup
```

Manual `-n` / `--nodes` input is a fallback for explicitly instructed or
non-discoverable network topologies, not the normal cluster workflow.

When using cluster mode with `--setup`:
- Container is built locally and copied to all worker nodes
- Model is downloaded locally and copied to all worker nodes

### Cluster-Only Recipes

`deepseek-v4.1-flash` is a top-level symlink to
[`4x-spark-cluster/deepseek-v4.1-flash.yaml`](4x-spark-cluster/deepseek-v4.1-flash.yaml).
The short name selects the four-Spark recipe, so a service template that passes
its instance name to `run-recipe.sh` can use `vllm@deepseek-v4.1-flash` directly.
It uses the recipe's `vllm-node-b12x` image tag. For three Sparks, pass
`recipes/3x-spark-cluster/deepseek-v4.1-flash.yaml` explicitly.

### DeepSeek V4.1 Flash 1017

`deepseek-v4.1-flash-1017` selects the new
`deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint on four Sparks through
[`4x-spark-cluster/deepseek-v4.1-flash-1017.yaml`](4x-spark-cluster/deepseek-v4.1-flash-1017.yaml).
The corresponding service instance is `vllm@deepseek-v4.1-flash-1017`.
For three Sparks, select
[`3x-spark-cluster/deepseek-v4.1-flash-1017.yaml`](3x-spark-cluster/deepseek-v4.1-flash-1017.yaml)
explicitly. The original recipes remain available.

```bash
# Four Sparks, using the saved cluster configuration
./run-recipe.sh deepseek-v4.1-flash-1017 --setup

# Three Sparks
./run-recipe.sh recipes/3x-spark-cluster/deepseek-v4.1-flash-1017.yaml --setup
```

These recipes pin the full 1,048,576-token context window, retain the original
B12X/DSpark settings and five-image request cap, and explicitly select `--tokenizer-mode deepseek_v41` because the new
checkpoint ships no Jinja chat template. Clients must request the full new
model ID above. See the [compatibility and qualification notes](../docs/tuning/deepseek-v4.1-flash-1017.md)
before deployment, including the requirement to use the new checkpoint's
matching Engram tables. Validation here covers generated commands; the new
weights have not been qualified on a live Spark cluster. The three-node
variant removes the original `0.5g` KV cap because current B12X source
predicts it is insufficient for 1M context; both variants size KV memory
from their existing GPU utilization setting. Startup still rejects an
insufficient budget rather than silently reducing the context.

### DeepSeek V4.1 Flash image count

Both the three- and four-Spark recipes accept up to five images per request.
This deliberately raises eugr's recipe cap of one to match
[omp 18.11.0's fallback image budget](https://github.com/can1357/oh-my-pi/blob/v18.11.0/packages/snapcompact/src/snapcompact.ts#L577-L587).
omp retains images from earlier turns and trims the oldest beyond that budget;
a server cap of one rejects the request as soon as a second image enters the
conversation. The count applies to the complete request, including history and
tool results, rather than just the latest user message.

The one-image cap is a serving choice, not a DS4.1F implementation limit:
[vLLM's model guide](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash#images)
documents support for multiple images without a model-specific count limit.
vLLM rejects requests above the configured cap; it does not trim them. Clients
without their own trimming must stay within five images.

The new cap takes effect after relaunching with the updated recipe; no image
rebuild is needed. Before replacing a client-side workaround, verify two- and
five-image requests and an omp conversation that accumulates more than five
images. Check memory use as well: recipe dry-runs do not validate vision
profiling, multi-image accuracy, or the running deployment's configuration.

Some models are too large to run on a single node. These recipes have `cluster_only: true` and will fail with a helpful error if you try to run them in solo mode:

```bash
$ ./run-recipe.sh minimax-m2-awq --solo
Error: Recipe 'MiniMax-M2-AWQ' requires cluster mode.
This model is too large to run on a single node.

Options:
  1. Auto-discover and save:  ./run-recipe.sh --discover
     Then run:                ./run-recipe.sh minimax-m2-awq
  2. If autodiscovery cannot support the topology, specify nodes as instructed.
```

## Setup Options

| Flag | Description |
|------|-------------|
| `--setup` | Full setup: build (if missing) + download (if missing) + run |
| `--build-only` | Only build/copy the container, don't run |
| `--download-only` | Only download/copy the model, don't run |
| `--force-build` | Rebuild even if container exists |
| `--force-download` | Re-download even if model exists |
| `--dry-run` | Show what would happen without executing |

## Recipe Format

```yaml
# Required fields
name: Human-readable name
container: docker-image-name
command: |
  vllm serve model/name \
      --port {port} \
      --host {host}

# Optional fields
description: What this recipe does
model: org/model-name              # HuggingFace model ID for --setup downloads
cluster_only: false                # Set to true if model requires cluster mode
build_args:                        # Extra args for build-and-copy.sh
  - --exp-mxfp4                    # e.g., for MXFP4 Dockerfile
mods:
  - mods/some-patch
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.85
  max_model_len: 32000
env:
  SOME_VAR: "value"
```

### Build Arguments

The `build_args` field passes flags to `build-and-copy.sh`:

| Flag | Description |
|------|-------------|
| `--exp-mxfp4` | Use MXFP4 Dockerfile (for MXFP4 quantized models) |
| `--exp-b12x` | Use the B12X image profile, which defaults to the `vllm-node-b12x` tag |
| `--use-wheels` | Build the runner image from prebuilt or local wheels instead of pulling `eugr/spark-vllm:latest` |

### Parameter Substitution

Use `{param_name}` in the command to substitute values from defaults or CLI overrides:

```yaml
defaults:
  port: 8000
  tensor_parallel: 2

command: |
  vllm serve my/model \
      --port {port} \
      -tp {tensor_parallel}
```

Override at runtime:
```bash
./run-recipe.sh my-recipe --port 9000 --tp 4
```

## CLI Reference

```
Usage: ./run-recipe.sh [OPTIONS] [RECIPE]

Cluster discovery:
  --discover                  Auto-detect cluster nodes and save to .env
  --show-env                  Show current .env configuration
  --config FILE               Path to .env configuration file (default: .env in repo directory)

Recipe overrides:
  --port PORT                 Override port
  --host HOST                 Override host
  --tensor-parallel, --tp N   Override tensor parallelism
  --gpu-memory-utilization N  Override GPU memory utilization (--gpu-mem)
  --max-model-len N           Override max model length

Setup options:
  --setup                     Full setup: build + download + run
  --build-only                Only build/copy container, don't run
  --download-only             Only download/copy model, don't run
  --force-build               Rebuild even if container exists
  --force-download            Re-download even if model exists

Launch options:
  --solo                      Run in solo mode (single node, no Ray)
  --ray                       Opt into Ray for multi-node vLLM
  --no-ray                    Default multi-node no-Ray mode (accepted for compatibility)
  -n, --nodes IPS             Manual fallback/override: comma-separated node IPs
                              (first = head)
  -d, --daemon                Run in daemon mode
  -t, --container IMAGE       Override container from recipe
  --name NAME                 Override container name
  --nccl-debug LEVEL          NCCL debug level (VERSION, WARN, INFO, TRACE)
  --apply-mod PATH            Apply an extra mod directory or zip (repeatable)
  --apply-vllm-pr PR_OR_URL   Apply a vLLM PR number or full GitHub PR URL
                              (repeatable, runtime-only, ordered with --apply-mod)
  -p, --publish HOST:CONTAINER
                              Publish a container port in solo mode (repeatable)
  -v, --volume LOCAL:CONTAINER
                              Map a volume using Docker syntax (repeatable)
  --master-port PORT          Cluster coordination port: Ray head port or PyTorch
                              distributed master port (default: 29501).
                              Alias: --head-port
  --eth-if IFACE              Override Ethernet interface
  --ib-if IFACE               Override InfiniBand interface
  -e VAR=VALUE                Pass environment variable to container (repeatable)
  -j N                        Number of parallel build jobs
  --no-cache-dirs             Do not mount ~/.cache/vllm, ~/.cache/flashinfer, ~/.cache/b12x, ~/.triton, ~/.tilelang
  --keep-entrypoint           Keep the Docker image entrypoint
  --earlyoom                  Run earlyoom as the container foreground process
  --earlyoom-args ARGS        Arguments passed to earlyoom
  --non-privileged            Run container without --privileged
  --mem-limit-gb N            Memory limit in GB (only with --non-privileged)
  --mem-swap-limit-gb N       Memory+swap limit in GB (only with --non-privileged)
  --pids-limit N              Process limit (only with --non-privileged)
  --shm-size-gb N             Shared memory size in GB (only with --non-privileged)

Extra vLLM arguments:
  -- ARGS...                  Pass additional arguments directly to vLLM

Other:
  --dry-run                   Show what would be executed
  --list, -l                  List available recipes
```

`--earlyoom` uses the same optional monitor as `launch-cluster.sh`. The default arguments are `-M 524288,102400 -s 100 -r 60`; override them with `--earlyoom-args "..."` or `VLLM_SPARK_EARLYOOM_ARGS`. `-M` values are KiB, so the default sends SIGTERM below 512 MiB available memory and SIGKILL below 100 MiB. For example:

```bash
./run-recipe.sh minimax-m2-awq --solo \
  --earlyoom --earlyoom-args "-M 786432,196608 -s 100 -r 120"
```

`--apply-vllm-pr` is a launch option, including when combined with `--setup`.
It creates an ephemeral runtime mod and does not pass the PR to the image build.
Bare numbers select PRs from `vllm-project/vllm`. A full public
`https://github.com/OWNER/REPO/pull/NUMBER` URL downloads the PR from the named
repository instead.
It accepts Python/package-only changes under `vllm/`; tests, docs, CI files,
and `setup.py` are ignored. The container must already provide dependencies
declared by an ignored `setup.py` change. PRs that modify native code or other
build/source-tree files must use `build-and-copy.sh --apply-vllm-pr` instead.
Recipe-declared mods are applied first, then command-line `--apply-mod` and
`--apply-vllm-pr` layers in their specified order:

```bash
./run-recipe.sh my-recipe --solo \
  --apply-mod mods/use-official-vllm \
  --apply-vllm-pr https://github.com/local-inference-lab/vllm/pull/669 \
  --apply-mod mods/my-followup-fix
```

## Environment Variables and Volumes

Both `run-recipe.sh` and `launch-cluster.sh` accept repeatable `-e VAR=VALUE`
container environment variables and repeatable `-v LOCAL:CONTAINER` Docker
volume mappings. Put `launch-cluster.sh` options before `exec`.

```bash
./run-recipe.sh my-recipe --solo \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models"

./launch-cluster.sh --solo \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models" \
  exec vllm serve /models/my-model --port 8000
```

In cluster mode, each volume mapping is applied to every node. Ensure the local
path exists with suitable contents and permissions on each host.

For a gated Hugging Face model, export the token for host-side downloads and
also pass it to the container:

```bash
export HF_TOKEN="YOUR_TOKEN"
./run-recipe.sh my-recipe -e HF_TOKEN="$HF_TOKEN" --setup
```

Avoid printing or sharing the expanded command because it contains the token.

## Extra vLLM Arguments

Use the Unix-style `--` separator to pass additional arguments directly to vLLM. Any arguments after `--` are appended verbatim to the vLLM command.

```bash
# Override load format
./run-recipe.sh my-recipe --solo -- --load-format safetensors

# Set a custom served model name
./run-recipe.sh my-recipe --solo -- --served-model-name my-api-name

# Configure CUDA graph mode
./run-recipe.sh my-recipe --solo -- -cc.cudagraph_mode=PIECEWISE

# Multiple extra arguments
./run-recipe.sh my-recipe --solo -- --load-format auto --enforce-eager --seed 42
```

These arguments are appended to the end of the generated vLLM command after all template substitutions. If an option occurs more than once, vLLM uses its last occurrence.

**Duplicate Detection**: If you pass an argument that conflicts with a CLI override (e.g., `--port` when you also used `--port`), the runner warns that the later extra argument wins. Prefer the built-in `--port`, `--host`, `--tp`, `--gpu-mem`, and `--max-model-len` overrides when they cover the setting you need.

## Creating a Recipe

1. Create a new `.yaml` file in `recipes/`
2. Specify required fields: `name`, `container`, `command`
3. Add `build_args` if your model needs special build options
4. Add `mods` if your model needs patches
5. Set `cluster_only: true` if model is too large for single node
6. Set sensible `defaults`
7. Add `env` variables if needed

Example:
```yaml
name: My Model
description: My custom model setup
container: vllm-node

# New recipes should use the default vllm-node image and omit legacy TF5 build args.

mods:
  - mods/my-fix

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.85

command: |
  vllm serve org/my-model \
      --port {port} \
      --host {host} \
      -tp {tensor_parallel} \
      --gpu-memory-utilization {gpu_memory_utilization}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  autodiscover.sh                                        │
│  - Interface detection (standard / mesh topology)       │
│  - GB10 peer verification via SSH                       │
│  - CLUSTER_NODES and COPY_HOSTS discovery               │
│  - Interactive .env save with per-node confirmation     │
└──────────────────────────┬──────────────────────────────┘
                           │ sourced by
                           ▼
┌─────────────────────────────────────────────────────────┐
│  run-recipe.sh / run-recipe.py                          │
│  - Parses YAML recipe                                   │
│  - Loads / triggers cluster discovery (--discover)      │
│  - Handles --setup (build + download + run)             │
│  - Generates launch script from template                │
│  - Applies CLI overrides                                │
└──────────┬────────────────────────┬─────────────────────┘
           │ calls (for build)      │ calls (for download)
           ▼                        ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  build-and-copy.sh   │  │  hf-download.sh               │
│  - Docker build      │  │  - HuggingFace model download │
│  - Copy to COPY_HOSTS│  │  - Rsync to COPY_HOSTS        │
└──────────────────────┘  └───────────────────────────────┘
           │
           │ then calls (for run)
           ▼
┌─────────────────────────────────────────────────────────┐
│  launch-cluster.sh                                      │
│  - Cluster orchestration                                │
│  - Container lifecycle (trimmed to required node count) │
│  - Mod application                                      │
│  - Launch script execution                              │
└─────────────────────────────────────────────────────────┘
```

This separation follows the Unix philosophy: `run-recipe.sh` provides convenience, while the underlying scripts remain focused on their specific tasks.
