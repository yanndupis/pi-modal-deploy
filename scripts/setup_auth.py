#!/usr/bin/env python3
"""Ensure .env has PI_MODAL_API_KEY, generating one if needed."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path


ENV_KEY = "PI_MODAL_API_KEY"
DEFAULT_ENV_PATH = Path(".env")
MODAL_SECRET_NAME = "pi-modal-api-key"
MODAL_SECRET_KEY = "SGLANG_API_KEY"
KEY_BYTES = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--force", action="store_true", help="Rotate the existing key")
    parser.add_argument(
        "--create-modal-secret",
        action="store_true",
        help="Create or update the Modal Secret using the generated or existing key",
    )
    return parser.parse_args()


def parse_env_lines(lines: list[str]) -> tuple[list[str], str | None]:
    existing_value: str | None = None
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue

        key, value = line.split("=", 1)
        if key.strip() == ENV_KEY:
            existing_value = value.strip().strip("'\"")
            continue

        output.append(line)

    return output, existing_value


def read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines(keepends=True)


def write_env(path: Path, lines: list[str], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines and lines[-1] and not lines[-1].endswith("\n"):
        lines[-1] = f"{lines[-1]}\n"
    lines.append(f"{ENV_KEY}={key}\n")
    path.write_text("".join(lines))
    os.chmod(path, 0o600)


def ensure_local_key(path: Path, *, force: bool) -> str:
    kept_lines, existing_value = parse_env_lines(read_env_lines(path))

    if existing_value and not force:
        print(f"{ENV_KEY} already exists in {path}; leaving it unchanged.")
        return existing_value

    key = secrets.token_urlsafe(KEY_BYTES)
    write_env(path, kept_lines, key)

    action = "Rotated" if existing_value else "Generated"
    print(f"{action} {ENV_KEY} in {path}.")
    return key


def create_modal_secret(key: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        prefix="pi-modal-secret-",
        suffix=".json",
    ) as handle:
        secret_path = Path(handle.name)
        json.dump({MODAL_SECRET_KEY: key}, handle)
        handle.write("\n")

    os.chmod(secret_path, 0o600)
    try:
        subprocess.run(
            [
                "uv",
                "run",
                "modal",
                "secret",
                "create",
                MODAL_SECRET_NAME,
                "--force",
                "--from-json",
                str(secret_path),
            ],
            check=True,
        )
    finally:
        secret_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    key = ensure_local_key(args.env_file, force=args.force)

    if args.create_modal_secret:
        create_modal_secret(key)
        print(f"Created or updated Modal Secret {MODAL_SECRET_NAME}.")
    else:
        print("Create or update the Modal Secret with:")
        print(f"  uv run python {Path(__file__).as_posix()} --create-modal-secret")


if __name__ == "__main__":
    main()
