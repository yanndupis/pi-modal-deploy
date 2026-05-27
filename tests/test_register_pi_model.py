import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "register_pi_model.py"
SPEC = importlib.util.spec_from_file_location("register_pi_model", MODULE_PATH)
register_pi_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(register_pi_model)


class RegisterPiModelTests(unittest.TestCase):
    def test_update_config_advertises_model_context(self) -> None:
        config = register_pi_model.update_config(
            {},
            provider_id="pi-modal",
            base_url="https://example.modal.run/v1/",
            api_key="test-key",
            api="openai-completions",
            model_id="Qwen/Qwen3.5-27B-FP8",
            model_name="Qwen 3.5 27B FP8 on Modal",
            context_window=131072,
            max_tokens=8192,
            reasoning=True,
            thinking_format="qwen-chat-template",
        )

        provider = config["providers"]["pi-modal"]
        model = provider["models"][0]

        self.assertEqual(provider["baseUrl"], "https://example.modal.run/v1")
        self.assertEqual(model["contextWindow"], 131072)
        self.assertEqual(model["maxTokens"], 8192)


if __name__ == "__main__":
    unittest.main()
