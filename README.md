# pi-modal-serve

Deploy a Modal-hosted SGLang endpoint and register it as a Pi model provider.

## Install The Skill In Pi

For reusable setup, install this repo as a Pi package:

```bash
git clone <this-repo-url>
cd pi-modal-serve
uv sync
uv run modal setup
pi install .
```

If you are developing the skill locally and do not want to install it yet, use
the ad-hoc form:

```bash
pi --provider <existing-provider> --model <existing-model> --skill .
```

For a shared GitHub repo, users can install directly:

```bash
pi install git:github.com/<owner>/pi-modal-serve
```

## Bootstrap With Pi

If you have not deployed a model yet, start Pi with any provider/model you
already have available. That temporary model runs the deployment skill; after
the deploy finishes, the skill registers the Modal endpoint as `pi-modal`.

After installing the skill, launch Pi with the temporary model:

```bash
pi --provider <existing-provider> --model <existing-model>
```

In Pi, invoke the skill explicitly:

```text
/skill:pi-modal-deploy set up auth, deploy the default Qwen model on Modal, register it in Pi, and verify it works. I have not deployed anything yet.
```

For a one-shot non-interactive run from a local checkout:

```bash
pi --provider <existing-provider> --model <existing-model> --skill . -p \
  "Use the pi-modal-deploy skill to set up auth, deploy the default Qwen model on Modal, register it in Pi, and verify it works. I have not deployed anything yet."
```

Once the flow finishes, Pi should default to:

- Provider: `pi-modal`
- Model: `Qwen/Qwen3.5-27B-FP8`
- Thinking: `off` by default, with `Shift+Tab` available in interactive Pi to cycle thinking levels

## Manual Commands

The skill runs these steps for you, but the core commands are:

```bash
uv run python scripts/setup_auth.py --create-modal-secret
uv run --env-file .env modal run server.py --timeout 1800 --prompt "Say ready."
uv run --env-file .env modal deploy server.py
uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT.modal.run/v1"
```
