"""Modal/SGLang OpenAI-compatible server for Pi.

Setup:
    uv run modal setup
    uv run python scripts/setup_auth.py --create-modal-secret

Deploy:
    uv run modal deploy server.py

Smoke test:
    uv run --env-file .env modal run server.py
    uv run --env-file .env modal run server.py --enable-thinking
    uv run --env-file .env modal run server.py --tool-test

Skip build-time DeepGEMM precompile while iterating:
    PI_MODAL_PRECOMPILE_DEEPGEMM=0 uv run modal run server.py --help
"""

import asyncio
import json
import os
import shlex
import subprocess
import time
from typing import Any

import modal


MINUTES = 60
PORT = 8000

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_MODEL_REVISION = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"

APP_NAME = os.environ.get("PI_MODAL_APP_NAME", "pi-modal-qwen3-6-27b-fp8")
MODEL_ID = os.environ.get("PI_MODAL_MODEL_ID", DEFAULT_MODEL_ID)
MODEL_REVISION = os.environ.get("PI_MODAL_MODEL_REVISION")
if MODEL_REVISION is None and MODEL_ID == DEFAULT_MODEL_ID:
    MODEL_REVISION = DEFAULT_MODEL_REVISION
SERVED_MODEL_NAME = os.environ.get("PI_MODAL_SERVED_MODEL_NAME", MODEL_ID)

GPU = os.environ.get("PI_MODAL_GPU", "H100!:1")
TP_SIZE = int(os.environ.get("PI_MODAL_TP_SIZE", "1"))
DEFAULT_MAX_MODEL_LEN = 131072
MAX_MODEL_LEN = int(os.environ.get("PI_MODAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))
SCALEDOWN_WINDOW = int(os.environ.get("PI_MODAL_SCALEDOWN_WINDOW", str(10 * MINUTES)))

TARGET_INPUTS = int(os.environ.get("PI_MODAL_TARGET_INPUTS", "8"))
MAX_INPUTS = int(os.environ.get("PI_MODAL_MAX_INPUTS", "32"))
MEM_FRACTION_STATIC = os.environ.get("PI_MODAL_MEM_FRACTION_STATIC", "0.8")
STARTUP_TIMEOUT = int(os.environ.get("PI_MODAL_STARTUP_TIMEOUT", str(20 * MINUTES)))
REQUEST_TIMEOUT = int(os.environ.get("PI_MODAL_REQUEST_TIMEOUT", str(60 * MINUTES)))

SGLANG_IMAGE_TAG = os.environ.get(
    "PI_MODAL_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.12.post1-cu130-runtime"
)
SGLANG_PYTHON = os.environ.get("PI_MODAL_SGLANG_PYTHON", "/usr/bin/python3.12")
AUTH_SECRET_NAME = os.environ.get("PI_MODAL_AUTH_SECRET", "pi-modal-api-key")
SGLANG_API_KEY_ENV = "SGLANG_API_KEY"
REASONING_PARSER = os.environ.get("PI_MODAL_REASONING_PARSER", "qwen3").strip()
TOOL_CALL_PARSER = os.environ.get("PI_MODAL_TOOL_CALL_PARSER", "qwen3_coder").strip()
EXTRA_SERVER_ARGS = os.environ.get("PI_MODAL_EXTRA_SERVER_ARGS", "")
PRECOMPILE_DEEPGEMM = os.environ.get("PI_MODAL_PRECOMPILE_DEEPGEMM", "1") != "0"

HF_CACHE_VOL = modal.Volume.from_name(
    os.environ.get("PI_MODAL_HF_CACHE_VOLUME", "huggingface-cache"),
    create_if_missing=True,
)
HF_CACHE_PATH = os.environ.get("PI_MODAL_HF_CACHE_PATH", "/cache/huggingface")

DG_CACHE_VOL = modal.Volume.from_name(
    os.environ.get("PI_MODAL_DEEPGEMM_CACHE_VOLUME", "deepgemm-cache"),
    create_if_missing=True,
)
DG_CACHE_PATH = os.environ.get("PI_MODAL_DEEPGEMM_CACHE_PATH", "/cache/deep_gemm")

AUTH_SECRET = modal.Secret.from_name(
    AUTH_SECRET_NAME,
    required_keys=[SGLANG_API_KEY_ENV],
)

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG, add_python="3.11")
    .entrypoint([])
    .run_commands(f"{SGLANG_PYTHON} -m pip install distro==1.9.0")
    .uv_pip_install("requests==2.32.5")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
            "SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
            "SGLANG_USE_IPC_POOL_HANDLE_CACHE": "1",
        }
    )
)


def compile_deep_gemm() -> None:
    if not int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        print("Skipping DeepGEMM precompile because SGLANG_ENABLE_JIT_DEEPGEMM=0.")
        return

    cmd = [
        SGLANG_PYTHON,
        "-m",
        "sglang.compile_deep_gemm",
        "--model-path",
        MODEL_ID,
        "--tp",
        str(TP_SIZE),
    ]
    if MODEL_REVISION:
        cmd.extend(["--revision", MODEL_REVISION])
    print("Precompiling DeepGEMM kernels:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


if PRECOMPILE_DEEPGEMM:
    sglang_image = sglang_image.run_function(
        compile_deep_gemm,
        volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
        gpu=GPU,
        timeout=REQUEST_TIMEOUT,
    )

app = modal.App(name=APP_NAME)


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _redact_command(cmd: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False

    for part in cmd:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue

        redacted.append(part)
        if part in {"--api-key", "--admin-api-key"}:
            hide_next = True

    return " ".join(shlex.quote(part) for part in redacted)


def _server_command(
    *,
    model_id: str,
    model_revision: str | None,
    served_model_name: str,
    max_model_len: int,
    reasoning_parser: str,
    tool_call_parser: str,
    api_key: str,
) -> list[str]:
    cmd = [
        SGLANG_PYTHON,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_id,
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp",
        str(TP_SIZE),
        "--context-length",
        str(max_model_len),
        "--api-key",
        api_key,
        "--enable-cache-report",
        "--enable-metrics",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
        "--cuda-graph-max-bs",
        str(max(2, TARGET_INPUTS * 2)),
        "--max-running-requests",
        str(MAX_INPUTS),
    ]

    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])

    if tool_call_parser:
        cmd.extend(["--tool-call-parser", tool_call_parser])

    if model_revision:
        cmd.extend(["--revision", model_revision])

    if EXTRA_SERVER_ARGS:
        cmd.extend(shlex.split(EXTRA_SERVER_ARGS))

    return cmd


def _check_running(process: subprocess.Popen[Any]) -> None:
    if (return_code := process.poll()) is not None:
        raise subprocess.CalledProcessError(return_code, process.args)


def _wait_ready(process: subprocess.Popen[Any], timeout: int = STARTUP_TIMEOUT) -> None:
    import requests

    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{PORT}/health"

    while time.time() < deadline:
        try:
            _check_running(process)
            requests.get(health_url, timeout=5).raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.Timeout,
        ):
            time.sleep(2)

    raise TimeoutError(f"SGLang server was not healthy within {timeout} seconds")


def _warmup(*, model: str, api_key: str) -> None:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 8,
        "temperature": 0,
    }

    requests.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json=payload,
        headers=_auth_headers(api_key),
        timeout=120,
    ).raise_for_status()


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    secrets=[AUTH_SECRET],
    timeout=REQUEST_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=2,
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class SGLangServer:
    model_id: str = modal.parameter(default=MODEL_ID)
    model_revision: str = modal.parameter(default=MODEL_REVISION or "")
    served_model_name: str = modal.parameter(default=SERVED_MODEL_NAME)
    max_model_len: int = modal.parameter(default=MAX_MODEL_LEN)
    reasoning_parser: str = modal.parameter(default=REASONING_PARSER)
    tool_call_parser: str = modal.parameter(default=TOOL_CALL_PARSER)

    @modal.enter()
    def start(self) -> None:
        api_key = os.environ[SGLANG_API_KEY_ENV]
        revision = self.model_revision or None
        cmd = _server_command(
            model_id=self.model_id,
            model_revision=revision,
            served_model_name=self.served_model_name,
            max_model_len=int(self.max_model_len),
            reasoning_parser=self.reasoning_parser,
            tool_call_parser=self.tool_call_parser,
            api_key=api_key,
        )

        print("Starting SGLang server:")
        print(_redact_command(cmd))

        self.process = subprocess.Popen(cmd, start_new_session=True)
        _wait_ready(self.process)
        _warmup(model=self.served_model_name, api_key=api_key)
        print("SGLang server is ready.")

    @modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
    def serve(self) -> None:
        pass

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return

        import signal

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def _sample_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a concise assistant running behind an OpenAI-compatible API.",
        },
        {"role": "user", "content": prompt},
    ]


def _tool_call_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Use the weather tool to look up the weather in Paris.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to look up.",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256,
    }


async def _post_chat_completion(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    import aiohttp

    headers = _auth_headers(api_key)
    deadline = time.time() + timeout

    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        while time.time() < deadline:
            try:
                async with session.post(
                    "/v1/chat/completions",
                    json=payload,
                    timeout=120,
                ) as response:
                    if response.status in {502, 503, 504}:
                        await asyncio.sleep(2)
                        continue
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(
                            f"Chat completion failed with HTTP {response.status}: {body}"
                        )
                    return await response.json()
            except asyncio.TimeoutError:
                await asyncio.sleep(2)

    raise TimeoutError(f"No chat completion response within {timeout} seconds")


def _local_api_key() -> str:
    api_key = os.environ.get("PI_MODAL_API_KEY") or os.environ.get(SGLANG_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            "Set PI_MODAL_API_KEY locally to the same value stored in the "
            f"Modal Secret `{AUTH_SECRET_NAME}` as `{SGLANG_API_KEY_ENV}`."
        )
    return api_key


@app.local_entrypoint()
async def main(
    prompt: str = "Say hello from Modal in one sentence.",
    tool_test: bool = False,
    timeout: int = 30 * MINUTES,
    enable_thinking: bool = False,
) -> None:
    api_key = _local_api_key()
    server = SGLangServer(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION or "",
        served_model_name=SERVED_MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        reasoning_parser=REASONING_PARSER,
        tool_call_parser=TOOL_CALL_PARSER,
    )
    url = await server.serve.get_web_url.aio()
    print(f"Testing {SERVED_MODEL_NAME} at {url}")

    payload: dict[str, Any]
    if tool_test:
        payload = _tool_call_payload(SERVED_MODEL_NAME)
    else:
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": _sample_messages(prompt),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "max_tokens": 128,
            "temperature": 0,
        }

    response = await _post_chat_completion(
        url=url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )

    message = response["choices"][0]["message"]
    print(json.dumps(message, indent=2))

    if tool_test and not message.get("tool_calls"):
        raise RuntimeError("Tool-call smoke test returned no structured tool_calls.")
