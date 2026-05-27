#!/usr/bin/env python3
"""Register a Modal-hosted OpenAI-compatible model in Pi's models.json."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"
DEFAULT_SETTINGS_JSON = Path.home() / ".pi" / "agent" / "settings.json"
DEFAULT_API_KEY_ENV = "PI_MODAL_API_KEY"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3.5-27B-FP8"
DEFAULT_MODEL_ID = (
    os.environ.get("PI_MODAL_SERVED_MODEL_NAME")
    or os.environ.get("PI_MODAL_MODEL_ID")
    or DEFAULT_QWEN_MODEL_ID
)
DEFAULT_MODEL_NAME = os.environ.get("PI_MODAL_MODEL_NAME") or (
    "Qwen 3.5 27B FP8 on Modal"
    if DEFAULT_MODEL_ID == DEFAULT_QWEN_MODEL_ID
    else DEFAULT_MODEL_ID
)
DEFAULT_CONTEXT_WINDOW = int(os.environ.get("PI_MODAL_MAX_MODEL_LEN", "131072"))
DEFAULT_MAX_TOKENS = int(os.environ.get("PI_MODAL_MAX_TOKENS", "8192"))
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")
THINKING_FORMATS = (
    "openai",
    "openrouter",
    "deepseek",
    "together",
    "zai",
    "qwen",
    "qwen-chat-template",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON)
    parser.add_argument("--settings-json", type=Path, default=DEFAULT_SETTINGS_JSON)
    parser.add_argument("--provider-id", default="pi-modal")
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--api-key",
        help=f"Provider API key. Defaults to ${DEFAULT_API_KEY_ENV} when omitted.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--context-window", type=positive_int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--max-tokens", type=positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api", default="openai-completions")
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thinking-format", choices=THINKING_FORMATS, default="qwen-chat-template")
    parser.add_argument("--set-default", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-thinking-level", choices=THINKING_LEVELS, default="off")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def resolve_api_key(api_key: str | None) -> str:
    resolved = api_key or os.environ.get(DEFAULT_API_KEY_ENV, "")
    resolved = resolved.strip()
    if not resolved:
        raise SystemExit(f"Set {DEFAULT_API_KEY_ENV} or pass --api-key.")
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open() as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return config


def update_settings(
    settings: dict[str, Any],
    *,
    provider_id: str,
    model_id: str,
    default_thinking_level: str,
) -> dict[str, Any]:
    settings.update(
        {
            "defaultProvider": provider_id,
            "defaultModel": model_id,
            "defaultThinkingLevel": default_thinking_level,
        }
    )
    return settings


def update_config(
    config: dict[str, Any],
    *,
    provider_id: str,
    base_url: str,
    api_key: str,
    api: str,
    model_id: str,
    model_name: str,
    context_window: int,
    max_tokens: int,
    reasoning: bool,
    thinking_format: str,
) -> dict[str, Any]:
    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("models.json key 'providers' must be a JSON object")

    provider = providers.setdefault(provider_id, {})
    if not isinstance(provider, dict):
        raise ValueError(f"models.json provider '{provider_id}' must be a JSON object")

    compat = provider.setdefault("compat", {})
    if not isinstance(compat, dict):
        raise ValueError(f"models.json provider '{provider_id}'.compat must be a JSON object")

    compat.update(
        {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "thinkingFormat": thinking_format,
        }
    )

    provider.update(
        {
            "baseUrl": base_url.rstrip("/"),
            "api": api,
            "apiKey": api_key,
            "compat": compat,
        }
    )

    models = provider.setdefault("models", [])
    if not isinstance(models, list):
        raise ValueError(f"models.json provider '{provider_id}'.models must be a list")

    model_entry = {
        "id": model_id,
        "name": model_name,
        "reasoning": reasoning,
        "contextWindow": context_window,
        "maxTokens": max_tokens,
    }

    for index, existing in enumerate(models):
        if isinstance(existing, dict) and existing.get("id") == model_id:
            models[index] = {**existing, **model_entry}
            break
    else:
        models.append(model_entry)

    return config


def atomic_write_json(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )

    try:
        with os.fdopen(fd, "w") as file:
            json.dump(config, file, indent=2)
            file.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        finally:
            raise


def main() -> None:
    args = parse_args()
    config = load_config(args.models_json)
    updated = update_config(
        config,
        provider_id=args.provider_id,
        base_url=args.base_url,
        api_key=resolve_api_key(args.api_key),
        api=args.api,
        model_id=args.model_id,
        model_name=args.model_name,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        thinking_format=args.thinking_format,
    )
    atomic_write_json(args.models_json, updated)

    if args.set_default:
        settings = load_config(args.settings_json)
        update_settings(
            settings,
            provider_id=args.provider_id,
            model_id=args.model_id,
            default_thinking_level=args.default_thinking_level,
        )
        atomic_write_json(args.settings_json, settings)
        print(
            f"Registered {args.model_id} under provider {args.provider_id} "
            f"and set it as Pi's default in {args.settings_json}"
        )
    else:
        print(f"Registered {args.model_id} under provider {args.provider_id} in {args.models_json}")


if __name__ == "__main__":
    main()
