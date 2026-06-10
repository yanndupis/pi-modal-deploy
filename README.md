# pi-modal-deploy

Deploy Qwen 3.6 27B FP8 or DeepSeek V4 Flash FP4 on Modal with SGLang, automatically register the endpoint as a Pi model provider, and start using it from Pi.

## Features

- Sets up `PI_MODAL_API_KEY`, stores it in a Modal Secret, and registers Pi with the same bearer token so the endpoint is not left unauthenticated.
- Registers the Modal OpenAI-compatible `/v1` endpoint in Pi with model metadata, context length, and thinking-format defaults.
- Uses Modal's low-latency SGLang pattern with `@modal.experimental.http_server`, regional proxy routing, and warmup checks.
- Enables SGLang EAGLE speculative decoding for the supported model presets.
- Reuses Modal Volumes for Hugging Face model weights and DeepGEMM compilation artifacts where enabled to reduce repeated startup work.

## Demo

Watch Pi deploy a Modal-hosted model, switch to it, and use it to generate a small HTML page:

[![Watch the pi-modal-deploy demo](assets/demo-thumbnail.jpg)](docs/demo.md)

## Install The Skill In Pi

Install directly from GitHub:

```bash
pi install git:github.com/yanndupis/pi-modal-deploy
```

For local development, install this checkout as a Pi package:

```bash
git clone https://github.com/yanndupis/pi-modal-deploy.git
cd pi-modal-deploy
uv sync
uv run modal setup
pi install .
```

## Bootstrap With Pi

After installing the skill, invoke it explicitly in Pi:

```text
/skill:pi-modal-deploy set up auth, deploy the default model on Modal, register it in Pi, and verify it works.
```

For a one-shot non-interactive run from a local checkout:

```bash
pi --skill . -p \
  "Use the pi-modal-deploy skill to set up auth, deploy the default model on Modal, register it in Pi, and verify it works."
```

Once the flow finishes, Pi should use:

- Provider: `pi-modal`
- Model: the Modal-hosted model registered by the skill
- Thinking: `off` by default, with `Shift+Tab` available in interactive Pi to cycle thinking levels

## Manual Setup

If you want to run the commands yourself instead of asking Pi to run the skill, use the default Qwen 3.6 preset:

```bash
uv run python scripts/setup_auth.py --create-modal-secret
uv run --env-file .env modal run server.py --timeout 1800 --prompt "Say ready."
uv run --env-file .env modal deploy server.py
uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT/v1"
```

Use DeepSeek V4 Flash instead by prefixing the Modal and registration commands with the model id:

```bash
PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
uv run --env-file .env modal run server.py --timeout 3600 --prompt "Say ready."

PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
uv run --env-file .env modal deploy server.py

PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT/v1"
```

The deployed endpoint is a Modal HTTP Server URL; copy the URL Modal prints and append `/v1` when registering it with Pi.

## Supported And Tested Models

This repo ships presets for the following Modal-hosted SGLang models:

| Model | Modal GPU | Context |
| --- | --- | ---: |
| `Qwen/Qwen3.6-27B-FP8` | `H100:1` | `131072` |
| `deepseek-ai/DeepSeek-V4-Flash` | `H200:4` | `65536` |

Qwen 3.6 is the default model. Both presets use SGLang EAGLE speculative decoding for lower latency. To use DeepSeek V4 Flash instead, set:

```bash
PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash
```

Both presets are intended for remote GPU serving on Modal, not local model execution.

## Cost And Startup Expectations

Modal bills for actual runtime, so cost depends mostly on how long the endpoint stays warm. At current Modal GPU pricing, the rough GPU-only cost is:

| Preset | GPU cost |
| --- | ---: |
| Qwen 3.6 on `H100:1` | about `$3.95 / hour` |
| DeepSeek V4 Flash on `H200:4` | about `$18.16 / hour` |

CPU, memory, volumes, region multipliers, and non-preemptible settings can add to this. Check the [Modal pricing page](https://modal.com/pricing) for current rates before running long sessions.

Cold starts can take several minutes because SGLang has to load model weights and prepare runtime kernels. DeepSeek V4 Flash can spend significant time on first-run CUDA graph capture. Once warm, short smoke-test requests have completed quickly in validation. For interactive Pi sessions, consider keeping the Modal app warm long enough to avoid repeated cold starts.

## References

- [Modal SGLang low-latency example](https://modal.com/docs/examples/sglang_low_latency)
- [SGLang Qwen3.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6)
- [SGLang DeepSeek V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
