# pi-modal-serve

Deploy a Modal-hosted SGLang endpoint and register it as a Pi model provider.

## Install The Skill In Pi

Install directly from GitHub:

```bash
pi install git:github.com/yanndupis/pi-modal-serve
```

For local development, install this checkout as a Pi package:

```bash
git clone https://github.com/yanndupis/pi-modal-serve.git
cd pi-modal-serve
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

## Manual Commands

The skill runs these steps for you, but the core commands are:

```bash
uv run python scripts/setup_auth.py --create-modal-secret
uv run --env-file .env modal run server.py --timeout 1800 --prompt "Say ready."
uv run --env-file .env modal deploy server.py
uv run --env-file .env python scripts/register_pi_model.py \
  --base-url "https://YOUR-ENDPOINT.modal.run/v1"
```
