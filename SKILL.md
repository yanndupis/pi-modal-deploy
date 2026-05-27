---
name: pi-modal-deploy
description: Deploy and manage a Modal-hosted SGLang OpenAI-compatible endpoint for Pi using this repository's server.py. Use when the user wants to deploy Qwen or another Hugging Face model on Modal, run Modal smoke tests, register the endpoint in Pi's ~/.pi/agent/models.json, debug Modal/SGLang/Pi integration, or update this repo's pi-modal deployment workflow.
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
- Context: `131072` by default
- Parser defaults: `PI_MODAL_REASONING_PARSER=qwen3`, `PI_MODAL_TOOL_CALL_PARSER=qwen3_coder`
- Auth: SGLang `--api-key` from Modal Secret `pi-modal-api-key`
- DeepGEMM: precompiled during Modal image build unless `PI_MODAL_PRECOMPILE_DEEPGEMM=0`

Treat these as a known-good starting point, not a product boundary. The workflow should support other Hugging Face models when the user supplies the model, revision, context length, GPU/TP shape, SGLang image, and parser settings that match that model.

`Qwen/Qwen3.6-27B-FP8` produced garbage output under `lmsysorg/sglang:v0.5.10.post1-cu130-runtime` because of SGLang issue #23687. Use an SGLang image that includes the Qwen3.6 FP8 loader fix, such as the default `v0.5.12.post1-cu130-runtime`, before trusting this model.

## Model And Context Tuning

Keep the server context length and the Pi model registration in sync. `server.py` reads `PI_MODAL_MAX_MODEL_LEN`, and `scripts/register_pi_model.py` uses the same environment variable as the default `contextWindow`.

The current default uses a 131,072-token context window for a single-H100 Modal deployment. If a selected model and GPU/TP configuration can support a larger window, set `PI_MODAL_MAX_MODEL_LEN` before smoke testing, deploying, and registering:

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
uv run --env-file .env modal run server.py --timeout 1800
```

Empty parser values omit the SGLang parser flags. Use model-specific parser names only when the selected model and SGLang image support them.

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
- `sglang.compile_deep_gemm` finishes if precompile is enabled.
- Server command includes the pinned revision.
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

Pass only if the returned message contains structured `tool_calls`. If the model emits plain text instead, inspect SGLang parser flags before changing Pi config. For the current Qwen default, those are:

- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_coder`
- `chat_template_kwargs: {"enable_thinking": false}` in smoke tests

## Environment Overrides

Use env vars for deliberate experiments:

```bash
PI_MODAL_APP_NAME=pi-modal-test \
PI_MODAL_MODEL_ID=Qwen/Qwen3.6-27B-FP8 \
PI_MODAL_MODEL_REVISION=e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
PI_MODAL_MAX_MODEL_LEN=131072 \
uv run --env-file .env modal run server.py --timeout 1800
```

If `PI_MODAL_MODEL_ID` differs from the default and `PI_MODAL_MODEL_REVISION` is unset, no revision is pinned. Set both for reproducible experiments. Override `PI_MODAL_REASONING_PARSER` and `PI_MODAL_TOOL_CALL_PARSER`, or set them to empty strings, when testing non-Qwen models.

## Debugging

- HTTP `401` or `403`: confirm `PI_MODAL_API_KEY` matches Modal Secret `SGLANG_API_KEY`.
- HTTP `502`, `503`, or `504`: the server may still be cold-starting; rerun with a longer timeout and inspect Modal logs.
- Garbage output: check for SGLang loader warnings and confirm the SGLang image includes the Qwen3.6 FP8 loader fix. Older `v0.5.10.post1` images produced garbage output with `Qwen/Qwen3.6-27B-FP8`.
- Slow first request: confirm DeepGEMM precompile ran during image build and HF cache used the pinned snapshot.
- Pi model not visible: reopen Pi's `/model` selector after updating `models.json`.

## Quality Bar

Before considering the skill's work done:

- `uv run python -m py_compile server.py scripts/register_pi_model.py scripts/setup_auth.py` passes.
- `uv run modal run server.py --help` parses.
- The Modal smoke test returns coherent content.
- Auth setup is tested against a temp `.env` file if the auth helper changed.
- Pi registration is tested against a temp JSON file or the user's real Pi config, depending on the request.
- Do not expose API keys in final answers or logs.
