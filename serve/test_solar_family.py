"""Solar-Open-100B through the server: its effort words, the template's globals and its message markers.

    python -m unittest serve.test_solar_family -v

No GPU, no pack, no model: the family's rules are pure functions and one unbound `Service.set_family`.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import frontend                      # noqa: E402
from serve import server                        # noqa: E402

SOLAR_IDS = {"<|begin|>": 20, "<|end|>": 21, "<|think|>": 22, "<|content|>": 23, "<|flush|>": 24}
SOLAR_SPECIALS = {"tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2}


class EffortWords(unittest.TestCase):
    def tearDown(self):
        frontend.use_family(None)

    def test_minimal_is_kept_and_marks_the_reply_as_content(self):
        frontend.use_family("glm4moe")
        # the template reads "minimal" (its empty think block); enable_thinking is the server's own flag that the
        # reply starts with the answer, so the answer is not reported as reasoning_content
        self.assertEqual(frontend.effort_kwargs("minimal"),
                         {"reasoning_effort": "minimal", "enable_thinking": False})
        self.assertEqual(frontend.effort_kwargs("low"),
                         {"reasoning_effort": "low", "enable_thinking": False})

    def test_none_becomes_the_same_empty_block(self):
        frontend.use_family("glm4moe")
        self.assertEqual(frontend.effort_kwargs("none"),
                         {"reasoning_effort": "minimal", "enable_thinking": False})

    def test_high_passes_through_without_the_content_flag(self):
        frontend.use_family("glm4moe")
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "high"})
        self.assertEqual(frontend.effort_kwargs("xhigh"), {"reasoning_effort": "high"})

    def test_the_qwen_table_is_unchanged(self):
        frontend.use_family(None)
        self.assertEqual(frontend.effort_kwargs("minimal"), {"enable_thinking": False})
        self.assertEqual(frontend.effort_kwargs("high"), {"reasoning_effort": "xhigh"})


class TemplateGlobals(unittest.TestCase):
    def test_a_real_template_renders_the_date(self):
        # llama.cpp provides strftime_now; Solar's template prints a date and every request failed without it
        import tempfile
        src = '{%- set now = strftime_now("%Y-%m-%d") -%}<{{ now }}>'
        with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False, encoding="utf-8") as f:
            f.write(src)
            path = f.name
        try:
            self.assertRegex(frontend.ChatTemplate(path).render([]), r"^<\d{4}-\d{2}-\d{2}>$")
        finally:
            Path(path).unlink(missing_ok=True)


class MessageMarkers(unittest.TestCase):
    def _svc(self):
        return types.SimpleNamespace(tok=types.SimpleNamespace(ids=dict(SOLAR_IDS), special_ids=dict(SOLAR_SPECIALS)),
                                     family=None, stop_ids=set(), skip_ids=set())

    def test_stop_and_skip_ids(self):
        svc = self._svc()
        server.Service.set_family(svc, "glm4moe")
        self.assertEqual(svc.stop_ids, {2, 20, 21})      # eos, <|begin|>, <|end|>
        self.assertEqual(svc.skip_ids, {22, 23, 24})     # <|think|>, <|content|>, <|flush|>

    def test_the_family_is_inferred_from_the_tokenizer(self):
        self.assertEqual(server.arch_from_tokenizer(None, "solar-open"), "glm4moe")
        self.assertEqual(server.arch_from_tokenizer("qwen4exp", "solar-open"), "qwen4exp")


if __name__ == "__main__":
    unittest.main()
