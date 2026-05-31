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

Check the Modal entrypoint without building a remote image:
    uv run modal run server.py --help
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

QWEN_MODEL_ID = "Qwen/Qwen3.6-27B-FP8"
QWEN_MODEL_REVISION = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
DEEPSEEK_V4_FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
QWEN_SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.12.post1-cu130-runtime"
DEEPSEEK_SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.12.post1"
DEFAULT_MODEL_ID = QWEN_MODEL_ID
QWEN_SPECULATIVE_EXTRA_SERVER_ARGS = " ".join(
    [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--page-size",
        "64",
    ]
)
DEEPSEEK_V4_FLASH_EXTRA_SERVER_ARGS = " ".join(
    [
        "--trust-remote-code",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--disable-flashinfer-autotune",
        "--max-total-tokens",
        "262144",
    ]
)

MODEL_PRESETS = {
    QWEN_MODEL_ID: {
        "app_name": "pi-modal-qwen3-6-27b-fp8",
        "extra_server_args": QWEN_SPECULATIVE_EXTRA_SERVER_ARGS,
        "gpu": "H100:1",
        "max_model_len": "131072",
        "mem_fraction_static": "0.8",
        "precompile_deepgemm": "1",
        "reasoning_parser": "qwen3",
        "revision": QWEN_MODEL_REVISION,
        "sglang_env": "SGLANG_ENABLE_SPEC_V2=1",
        "sglang_image": QWEN_SGLANG_IMAGE_TAG,
        "thinking_template_flag": "enable_thinking",
        "tool_call_parser": "qwen3_coder",
        "tp_size": "1",
    },
    DEEPSEEK_V4_FLASH_MODEL_ID: {
        "app_name": "pi-modal-deepseek-v4-flash",
        "extra_server_args": DEEPSEEK_V4_FLASH_EXTRA_SERVER_ARGS,
        "gpu": "H200:4",
        "max_model_len": "65536",
        "mem_fraction_static": "",
        "precompile_deepgemm": "0",
        "reasoning_parser": "deepseek-v4",
        "sglang_env": "SGLANG_ENABLE_JIT_DEEPGEMM=0",
        "sglang_image": DEEPSEEK_SGLANG_IMAGE_TAG,
        "thinking_template_flag": "thinking",
        "tool_call_parser": "deepseekv4",
        "tp_size": "4",
        "warmup_timeout": "600",
    },
}


def _parse_env_assignments(value: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for assignment in shlex.split(value):
        key, separator, env_value = assignment.partition("=")
        if not separator or not key:
            raise ValueError(
                "PI_MODAL_SGLANG_ENV must be a space-separated list of KEY=VALUE pairs."
            )
        env[key] = env_value
    return env


MODEL_ID = os.environ.get("PI_MODAL_MODEL_ID", DEFAULT_MODEL_ID)
MODEL_PRESET = MODEL_PRESETS.get(MODEL_ID, {})
MODEL_REVISION = os.environ.get("PI_MODAL_MODEL_REVISION")
if MODEL_REVISION is None:
    MODEL_REVISION = MODEL_PRESET.get("revision")
SERVED_MODEL_NAME = os.environ.get(
    "PI_MODAL_SERVED_MODEL_NAME", MODEL_PRESET.get("served_model_name", MODEL_ID)
)

APP_NAME = os.environ.get(
    "PI_MODAL_APP_NAME", MODEL_PRESET.get("app_name", "pi-modal-sglang")
)
GPU = os.environ.get("PI_MODAL_GPU", MODEL_PRESET.get("gpu", "H100:1"))
TP_SIZE = int(os.environ.get("PI_MODAL_TP_SIZE", MODEL_PRESET.get("tp_size", "1")))
DEFAULT_MAX_MODEL_LEN = int(MODEL_PRESET.get("max_model_len", "131072"))
MAX_MODEL_LEN = int(
    os.environ.get("PI_MODAL_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
)
SCALEDOWN_WINDOW = int(
    os.environ.get("PI_MODAL_SCALEDOWN_WINDOW", str(10 * MINUTES))
)

TARGET_INPUTS = int(os.environ.get("PI_MODAL_TARGET_INPUTS", "8"))
MAX_INPUTS = int(os.environ.get("PI_MODAL_MAX_INPUTS", "32"))
MAX_CONTAINERS = int(os.environ.get("PI_MODAL_MAX_CONTAINERS", "1"))
MEM_FRACTION_STATIC = os.environ.get(
    "PI_MODAL_MEM_FRACTION_STATIC", MODEL_PRESET.get("mem_fraction_static", "")
)
STARTUP_TIMEOUT = int(
    os.environ.get("PI_MODAL_STARTUP_TIMEOUT", str(20 * MINUTES))
)
REQUEST_TIMEOUT = int(os.environ.get("PI_MODAL_REQUEST_TIMEOUT", str(60 * MINUTES)))
WARMUP_TIMEOUT = int(
    os.environ.get("PI_MODAL_WARMUP_TIMEOUT", MODEL_PRESET.get("warmup_timeout", "120"))
)

SGLANG_IMAGE_TAG = os.environ.get(
    "PI_MODAL_SGLANG_IMAGE",
    MODEL_PRESET.get("sglang_image", QWEN_SGLANG_IMAGE_TAG),
)
SGLANG_PYTHON = os.environ.get("PI_MODAL_SGLANG_PYTHON", "/usr/bin/python3.12")
AUTH_SECRET_NAME = os.environ.get("PI_MODAL_AUTH_SECRET", "pi-modal-api-key")
SGLANG_API_KEY_ENV = "SGLANG_API_KEY"
REASONING_PARSER = os.environ.get(
    "PI_MODAL_REASONING_PARSER",
    MODEL_PRESET.get("reasoning_parser", ""),
).strip()
TOOL_CALL_PARSER = os.environ.get(
    "PI_MODAL_TOOL_CALL_PARSER",
    MODEL_PRESET.get("tool_call_parser", ""),
).strip()
EXTRA_SERVER_ARGS = os.environ.get(
    "PI_MODAL_EXTRA_SERVER_ARGS", MODEL_PRESET.get("extra_server_args", "")
)
SGLANG_EXTRA_ENV = _parse_env_assignments(
    os.environ.get("PI_MODAL_SGLANG_ENV", MODEL_PRESET.get("sglang_env", ""))
)
THINKING_TEMPLATE_FLAG = os.environ.get(
    "PI_MODAL_THINKING_TEMPLATE_FLAG",
    MODEL_PRESET.get("thinking_template_flag", "enable_thinking"),
).strip()
PRECOMPILE_DEEPGEMM = (
    os.environ.get(
        "PI_MODAL_PRECOMPILE_DEEPGEMM",
        MODEL_PRESET.get("precompile_deepgemm", "0"),
    )
    != "0"
)

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

SGLANG_IMAGE_ENV = {
    "HF_HUB_CACHE": HF_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
    "SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
    "SGLANG_USE_IPC_POOL_HANDLE_CACHE": "1",
}
SGLANG_IMAGE_ENV.update(SGLANG_EXTRA_ENV)

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG, add_python="3.11")
    .entrypoint([])
    .run_commands(f"{SGLANG_PYTHON} -m pip install distro==1.9.0")
    .uv_pip_install("requests==2.32.5")
    .env(SGLANG_IMAGE_ENV)
)


def compile_deep_gemm() -> None:
    sglang_env = {**os.environ, **SGLANG_EXTRA_ENV}
    if not int(sglang_env.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
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
    subprocess.run(cmd, check=True, env=sglang_env)


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
    tp_size: int,
    mem_fraction_static: str,
    cuda_graph_max_bs: int,
    max_running_requests: int,
    reasoning_parser: str,
    tool_call_parser: str,
    extra_server_args: str,
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
        str(tp_size),
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
        "--cuda-graph-max-bs",
        str(cuda_graph_max_bs),
        "--max-running-requests",
        str(max_running_requests),
    ]

    if mem_fraction_static:
        cmd.extend(["--mem-fraction-static", mem_fraction_static])

    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])

    if tool_call_parser:
        cmd.extend(["--tool-call-parser", tool_call_parser])

    if model_revision:
        cmd.extend(["--revision", model_revision])

    if extra_server_args:
        cmd.extend(shlex.split(extra_server_args))

    return cmd


def _check_running(process: subprocess.Popen[Any]) -> None:
    if (return_code := process.poll()) is not None:
        raise subprocess.CalledProcessError(return_code, process.args)


def _check_model_info(*, url: str, api_key: str) -> None:
    import requests

    requests.get(
        url,
        headers=_auth_headers(api_key),
        timeout=5,
    ).raise_for_status()


def _wait_ready(
    process: subprocess.Popen[Any],
    *,
    api_key: str,
    timeout: int = STARTUP_TIMEOUT,
) -> None:
    import requests

    deadline = time.time() + timeout
    health_url = f"http://127.0.0.1:{PORT}/health"
    model_info_url = f"http://127.0.0.1:{PORT}/model_info"

    while time.time() < deadline:
        try:
            _check_running(process)
            health = requests.get(health_url, timeout=5)
            if health.ok:
                return
            if health.status_code == 503:
                _check_model_info(url=model_info_url, api_key=api_key)
                return
            health.raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            try:
                _check_running(process)
                _check_model_info(url=model_info_url, api_key=api_key)
                return
            except (
                subprocess.CalledProcessError,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
            ):
                time.sleep(2)
        except (
            subprocess.CalledProcessError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(2)

    raise TimeoutError(f"SGLang server was not healthy within {timeout} seconds")


def _warmup(*, model: str, api_key: str, timeout: int) -> None:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    if chat_template_kwargs := _chat_template_kwargs(False):
        payload["chat_template_kwargs"] = chat_template_kwargs

    requests.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json=payload,
        headers=_auth_headers(api_key),
        timeout=timeout,
    ).raise_for_status()


def _chat_template_kwargs(enable_thinking: bool) -> dict[str, bool]:
    if not THINKING_TEMPLATE_FLAG:
        return {}
    return {THINKING_TEMPLATE_FLAG: enable_thinking}


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    secrets=[AUTH_SECRET],
    timeout=REQUEST_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class SGLangServer:
    model_id: str = modal.parameter(default=MODEL_ID)
    model_revision: str = modal.parameter(default=MODEL_REVISION or "")
    served_model_name: str = modal.parameter(default=SERVED_MODEL_NAME)
    max_model_len: int = modal.parameter(default=MAX_MODEL_LEN)
    tp_size: int = modal.parameter(default=TP_SIZE)
    mem_fraction_static: str = modal.parameter(default=MEM_FRACTION_STATIC)
    cuda_graph_max_bs: int = modal.parameter(default=max(2, TARGET_INPUTS * 2))
    max_running_requests: int = modal.parameter(default=MAX_INPUTS)
    reasoning_parser: str = modal.parameter(default=REASONING_PARSER)
    tool_call_parser: str = modal.parameter(default=TOOL_CALL_PARSER)
    extra_server_args: str = modal.parameter(default=EXTRA_SERVER_ARGS)
    warmup_timeout: int = modal.parameter(default=WARMUP_TIMEOUT)

    @modal.enter()
    def start(self) -> None:
        api_key = os.environ[SGLANG_API_KEY_ENV]
        revision = self.model_revision or None
        cmd = _server_command(
            model_id=self.model_id,
            model_revision=revision,
            served_model_name=self.served_model_name,
            max_model_len=int(self.max_model_len),
            tp_size=int(self.tp_size),
            mem_fraction_static=self.mem_fraction_static,
            cuda_graph_max_bs=int(self.cuda_graph_max_bs),
            max_running_requests=int(self.max_running_requests),
            reasoning_parser=self.reasoning_parser,
            tool_call_parser=self.tool_call_parser,
            extra_server_args=self.extra_server_args,
            api_key=api_key,
        )

        print("Starting SGLang server:")
        print(_redact_command(cmd))
        if SGLANG_EXTRA_ENV:
            print("SGLang env override keys:", ", ".join(sorted(SGLANG_EXTRA_ENV)))

        self.process = subprocess.Popen(
            cmd,
            start_new_session=True,
            env={**os.environ, **SGLANG_EXTRA_ENV},
        )
        _wait_ready(self.process, api_key=api_key)
        _warmup(
            model=self.served_model_name,
            api_key=api_key,
            timeout=int(self.warmup_timeout),
        )
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
    payload: dict[str, Any] = {
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
        "max_tokens": 256,
    }

    if chat_template_kwargs := _chat_template_kwargs(False):
        payload["chat_template_kwargs"] = chat_template_kwargs

    return payload


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
        tp_size=TP_SIZE,
        mem_fraction_static=MEM_FRACTION_STATIC,
        cuda_graph_max_bs=max(2, TARGET_INPUTS * 2),
        max_running_requests=MAX_INPUTS,
        reasoning_parser=REASONING_PARSER,
        tool_call_parser=TOOL_CALL_PARSER,
        extra_server_args=EXTRA_SERVER_ARGS,
        warmup_timeout=WARMUP_TIMEOUT,
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
            "max_tokens": 128,
            "temperature": 0,
        }
        if chat_template_kwargs := _chat_template_kwargs(enable_thinking):
            payload["chat_template_kwargs"] = chat_template_kwargs

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
