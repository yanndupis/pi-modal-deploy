---
name: pi-modal-deploy
description: Deploy and manage Modal-hosted SGLang OpenAI-compatible endpoints for Pi using this repository's server.py. Use when the user wants to deploy Qwen3.6 27B FP8 or DeepSeek-V4-Flash FP4 on Modal, run Modal smoke tests, register the endpoint in Pi's ~/.pi/agent/models.json, debug Modal/SGLang/Pi integration, or update this repo's pi-modal deployment workflow.
---

# Pi Modal Deploy

Use this skill to deploy this repo's SGLang server on Modal and register it as a Pi model provider. Keep the workflow inspectable: run uv and Modal commands directly, report the endpoint URL, and preserve unrelated Pi config.

## Bootstrap From Pi

If the user has not deployed a Modal model yet, Pi must start with any already-working model/provider and this skill installed or loaded:

```bash
pi install .
pi --provider <existing-provider> --model <existing-model>
```

Use `pi install .` for reusable local setup. For ad-hoc development from a checkout, skip installation and launch with `pi --provider <existing-provider> --model <existing-model> --skill .` instead. In that bootstrapped Pi session, follow this skill to deploy the Modal endpoint, register provider `pi-modal`, and verify a plain `pi -p` prompt works. After registration, Pi can use the Modal-hosted model as its default.

## Current Defaults

- Current default model: `Qwen/Qwen3.6-27B-FP8`
- Pinned revision: `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`
- Engine: SGLang via `server.py`
- SGLang image: `lmsysorg/sglang:v0.5.12.post1-cu130-runtime`
- GPU: `H100!:1`
- Tensor parallelism: `1`
- Context: `131072` by default
- Parser defaults: `PI_MODAL_REASONING_PARSER=qwen3`, `PI_MODAL_TOOL_CALL_PARSER=qwen3_coder`
- Auth: SGLang `--api-key` from Modal Secret `pi-modal-api-key`
- DeepGEMM: precompiled during Modal image build unless `PI_MODAL_PRECOMPILE_DEEPGEMM=0`

Treat Qwen3.6 as the default because it is the validated single-GPU path. DeepSeek-V4-Flash FP4 is also supported through an explicit preset.

## Supported Model Presets

### Qwen 3.6 27B FP8

Use the default Qwen preset when the user wants the lowest-friction Modal deployment:

```bash
uv run --env-file .env modal run server.py --timeout 1800 --prompt "Say ready."
```

The default preset expands to:

- `PI_MODAL_APP_NAME=pi-modal-qwen3-6-27b-fp8`
- `PI_MODAL_MODEL_ID=Qwen/Qwen3.6-27B-FP8`
- `PI_MODAL_MODEL_REVISION=e89b16ebf1988b3d6befa7de50abc2d76f26eb09`
- `PI_MODAL_GPU=H100!:1`
- `PI_MODAL_TP_SIZE=1`
- `PI_MODAL_SGLANG_IMAGE=lmsysorg/sglang:v0.5.12.post1-cu130-runtime`
- `PI_MODAL_REASONING_PARSER=qwen3`
- `PI_MODAL_TOOL_CALL_PARSER=qwen3_coder`
- `PI_MODAL_THINKING_TEMPLATE_FLAG=enable_thinking`

### DeepSeek V4 Flash FP4

Use DeepSeek V4 Flash FP4 when the user wants the larger supported Modal-hosted model from this skill:

```bash
PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
uv run --env-file .env modal run server.py --timeout 3600 --prompt "Say ready."
```

This preset was smoke-tested on Modal `H200:4` with SGLang `v0.5.12.post1`. The validation returned HTTP `200 OK` from `/v1/chat/completions` with `content: "ready"` after the server reported ready.

The SGLang cookbook describes DeepSeek-V4-Flash as a 284B MoE model with 13B active parameters and single-node serving on 4 GPUs for B200, GB200, GB300, or H200. Modal supports `B200`, `H200`, and `H100` multi-GPU containers.

Default to `H200:4` for Modal because it is the lowest-cost official 4-GPU path in Modal's current pricing. A single H100 is not enough for this FP4 checkpoint. SGLang's generator includes an H100 FP4 path, but it uses `TP=8`, so it is both a larger GPU request and more expensive on Modal than `H200:4`.

If the smoke test passes, deploy and register with the same model environment:

```bash
PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
uv run --env-file .env modal deploy server.py

PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT.modal.run/v1"
```

The DeepSeek preset expands to:

- `PI_MODAL_APP_NAME=pi-modal-deepseek-v4-flash`
- `PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash`
- `PI_MODAL_GPU=H200:4`
- `PI_MODAL_TP_SIZE=4`
- `PI_MODAL_MAX_MODEL_LEN=8192`
- `PI_MODAL_SGLANG_IMAGE=lmsysorg/sglang:v0.5.12.post1`
- `PI_MODAL_REASONING_PARSER=deepseek-v4`
- `PI_MODAL_TOOL_CALL_PARSER=deepseekv4`
- `PI_MODAL_SGLANG_ENV="SGLANG_ENABLE_JIT_DEEPGEMM=0"`
- `PI_MODAL_THINKING_TEMPLATE_FLAG=thinking`
- `PI_MODAL_EXTRA_SERVER_ARGS="--trust-remote-code --moe-runner-backend flashinfer_mxfp4 --disable-cuda-graph --skip-server-warmup --disable-flashinfer-autotune --max-total-tokens 262144"`

Start the DeepSeek preset at `PI_MODAL_MAX_MODEL_LEN=8192` because it matches the upstream SGLang bring-up shape for a basic TP=4 server and keeps the first Modal validation conservative. DeepSeek V4 advertises a 1M-token context, and SGLang recommends much larger limits for long-reasoning benchmarks, but raise Modal context gradually after the base deployment works and GPU memory is confirmed.

Source notes for this preset:

- DeepSeek V4 Flash: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- Qwen3.6: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6
- Modal GPU shapes: https://modal.com/docs/guide/gpu
- Modal pricing: https://modal.com/pricing

## Model And Context Tuning

Keep the server context length and the Pi model registration in sync. `server.py` reads `PI_MODAL_MAX_MODEL_LEN`, and `scripts/register_pi_model.py` uses the same environment variable as the default `contextWindow` when it is set. Without an override, registration uses the selected model preset's context window.

Qwen3.6 defaults to 131,072 tokens. DeepSeek V4 Flash starts at 8,192 tokens for bring-up. If the deployed model and GPU shape can support a larger window, set `PI_MODAL_MAX_MODEL_LEN` before smoke testing, deploying, and registering:

```bash
PI_MODAL_MAX_MODEL_LEN=262144 uv run --env-file .env modal run server.py \
  --timeout 1800 --prompt "Say ready."
PI_MODAL_MAX_MODEL_LEN=262144 uv run --env-file .env modal deploy server.py
PI_MODAL_MAX_MODEL_LEN=262144 uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT.modal.run/v1"
```

For non-default models, set the model-related environment variables deliberately:

```bash
PI_MODAL_MODEL_ID=org/model-name \
PI_MODAL_MODEL_REVISION= \
PI_MODAL_SERVED_MODEL_NAME=org/model-name \
PI_MODAL_MAX_MODEL_LEN=131072 \
PI_MODAL_REASONING_PARSER= \
PI_MODAL_TOOL_CALL_PARSER= \
PI_MODAL_SGLANG_ENV= \
PI_MODAL_EXTRA_SERVER_ARGS= \
uv run --env-file .env modal run server.py --timeout 1800
```

Empty parser values omit the SGLang parser flags. Use model-specific parser names, `PI_MODAL_SGLANG_ENV`, and `PI_MODAL_EXTRA_SERVER_ARGS` only when the selected model and SGLang image support them.

## Preconditions

Confirm these before deploying:

```bash
uv run modal --version
uv run modal profile current
```

The user must have:

- Modal authenticated locally.
- A local `.env` with `PI_MODAL_API_KEY=<same key stored in Modal>`.
- A Modal Secret named `pi-modal-api-key` with key `SGLANG_API_KEY`.

If `.env` has no `PI_MODAL_API_KEY`, generate one:

```bash
uv run python scripts/setup_auth.py
```

To rotate the local key deliberately:

```bash
uv run python scripts/setup_auth.py --force
```

If the Modal Secret is missing, create it with:

```bash
uv run python scripts/setup_auth.py --create-modal-secret
```

To rotate both the local key and the Modal Secret deliberately:

```bash
uv run python scripts/setup_auth.py --force --create-modal-secret
```

## Smoke Test

Before a real deploy, run:

```bash
uv run --env-file .env modal run server.py --timeout 1800 --prompt "Say ready."
```

Success looks like:

- Modal image build succeeds.
- Server command includes the selected model and matching parser flags.
- For DeepSeek, server command includes `--moe-runner-backend flashinfer_mxfp4` and `--max-total-tokens 262144`.
- Response is HTTP `200 OK`.
- JSON message has coherent `content`.

For faster local CLI iteration where image precompile is irrelevant:

```bash
PI_MODAL_PRECOMPILE_DEEPGEMM=0 uv run modal run server.py --help
```

## Deploy

Deploy the app after the smoke test passes:

```bash
uv run --env-file .env modal deploy server.py
```

Capture the deployed web endpoint from Modal output. The Pi provider `baseUrl` should be the endpoint plus `/v1`.

## Register With Pi

Use the bundled updater instead of hand-editing JSON:

```bash
uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT.modal.run/v1"
```

The script atomically updates `~/.pi/agent/models.json`, creates or updates provider `pi-modal`, reads `PI_MODAL_API_KEY` from the environment unless `--api-key` is passed explicitly, advertises `contextWindow` from `PI_MODAL_MAX_MODEL_LEN`, and preserves unrelated providers. It also sets Pi's default provider/model and `defaultThinkingLevel: "off"` so reasoning-capable models return normal text through Pi by default.

Use a custom path for dry runs:

```bash
uv run python scripts/register_pi_model.py \
  --models-json /tmp/models.json \
  --settings-json /tmp/settings.json \
  --base-url "https://example.modal.run/v1" \
  --api-key test-key
```

## Tool-Call Validation

Before trusting the model in Pi, run:

```bash
uv run --env-file .env modal run server.py --timeout 1800 --tool-test
```

Pass only if the returned message contains structured `tool_calls`. If the model emits plain text instead, inspect SGLang parser flags before changing Pi config.

For Qwen3.6, those are:

- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_coder`
- `chat_template_kwargs: {"enable_thinking": false}` in smoke tests

For DeepSeek V4 Flash, those are:

- `--reasoning-parser deepseek-v4`
- `--tool-call-parser deepseekv4`
- `chat_template_kwargs: {"thinking": false}` in smoke tests

## Environment Overrides

Use env vars for deliberate experiments:

```bash
PI_MODAL_APP_NAME=pi-modal-test \
PI_MODAL_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash \
PI_MODAL_GPU=B200:4 \
PI_MODAL_MAX_MODEL_LEN=131072 \
uv run --env-file .env modal run server.py --timeout 1800
```

If `PI_MODAL_MODEL_ID` differs from the default and `PI_MODAL_MODEL_REVISION` is unset, no revision is pinned. Set both for reproducible experiments. Override `PI_MODAL_REASONING_PARSER` and `PI_MODAL_TOOL_CALL_PARSER`, or set them to empty strings, when testing unsupported models.

Use `PI_MODAL_SGLANG_ENV` for SGLang runtime environment flags that cannot be expressed as CLI arguments:

```bash
PI_MODAL_SGLANG_ENV="SGLANG_ENABLE_SPEC_V2=1" \
uv run --env-file .env modal run server.py --timeout 1800
```

## Debugging

- HTTP `401` or `403`: confirm `PI_MODAL_API_KEY` matches Modal Secret `SGLANG_API_KEY`.
- HTTP `502`, `503`, or `504`: the server may still be cold-starting; rerun with a longer timeout and inspect Modal logs.
- Modal logs that show `SIGTERM`, scheduler exit code `-15`, and `Runner terminated` after `POST /v1/chat/completions -> 200 OK` are normal teardown after `modal run` completes.
- Unexpected model output: confirm the selected model, revision, SGLang image, context length, and parser settings match each other.
- Slow first request: confirm the HF cache is populated and compare `H200:4` with `B200:4` if latency matters more than cost.
- Pi model not visible: reopen Pi's `/model` selector after updating `models.json`.

## Quality Bar

Before considering the skill's work done:

- `uv run python -m py_compile server.py scripts/register_pi_model.py scripts/setup_auth.py` passes.
- `uv run modal run server.py --help` parses.
- The Modal smoke test returns coherent content.
- Auth setup is tested against a temp `.env` file if the auth helper changed.
- Pi registration is tested against a temp JSON file or the user's real Pi config, depending on the request.
- Do not expose API keys in final answers or logs.
