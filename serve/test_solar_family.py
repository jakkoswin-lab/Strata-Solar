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

SOLAR_IDS = {"<|begin|>": 20, "<|end|>": 21, "<|think|>": 22, "<|content|>": 23, "<|flush|>": 24,
             "<|calls|>": 25, "<|tool_calls|>": 30}
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
        self.assertEqual(svc.stop_ids, {2, 20, 21, 25})   # eos, <|begin|>, <|end|>, <|calls|>
        self.assertEqual(svc.skip_ids, {22, 23, 24, 30})  # <|think|>, <|content|>, <|flush|>, <|tool_calls|>
        self.assertIs(svc.parser_cls, frontend.SolarOutputParser)

    def test_the_family_is_inferred_from_the_tokenizer(self):
        self.assertEqual(server.arch_from_tokenizer(None, "solar-open"), "glm4moe")
        self.assertEqual(server.arch_from_tokenizer("qwen4exp", "solar-open"), "qwen4exp")


OPEN, NAME, ARGS, END = ("<|tool_call:begin|>", "<|tool_call:name|>", "<|tool_call:args|>",
                         "<|tool_call:end|>")
CALL = OPEN + "abc123" + NAME + "get_weather" + ARGS + '{"city": "Seoul"}' + END
TOOLS = [{"name": "get_weather", "description": "weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]


class SolarToolCalls(unittest.TestCase):
    def parser(self, tools=TOOLS, **kw):
        return frontend.SolarOutputParser(tools=tools, **kw)

    def test_one_call_in_one_piece(self):
        p = self.parser()
        out = p.feed(CALL)
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.name, "get_weather")
        self.assertEqual(calls[0].call.arguments, {"city": "Seoul"})
        self.assertEqual(calls[0].call.id, "abc123")
        self.assertEqual("".join(e.text for e in out if e.kind == "content"), "")
        self.assertEqual(p.feed(""), [])

    def test_markers_split_across_tokens_do_not_leak(self):
        p = self.parser()
        out = []
        for ch in CALL + " done.":
            out += p.feed(ch)
        self.assertEqual([e.kind for e in out if e.kind == "tool_call"], ["tool_call"])
        self.assertEqual(out[0].call.name, "get_weather")
        self.assertEqual("".join(e.text for e in out if e.kind == "content"), " done.")

    def test_text_around_a_call(self):
        p = self.parser()
        out = p.feed("before " + CALL + " after")
        self.assertEqual([(e.kind, e.text) for e in out if e.kind == "content"],
                         [("content", "before "), ("content", " after")])
        self.assertEqual(len([e for e in out if e.kind == "tool_call"]), 1)

    def test_an_undeclared_name_stays_text(self):
        p = self.parser()                                  # only get_weather is declared
        out = p.feed(CALL.replace("get_weather", "delete_everything"))
        self.assertEqual([e.kind for e in out], ["content"])
        self.assertEqual(p.unread, 1)
        self.assertEqual(p.unread_names, ["delete_everything"])

    def test_bad_arguments_stay_text(self):
        p = self.parser()
        out = p.feed(OPEN + "x" + NAME + "get_weather" + ARGS + "{not json" + END)
        self.assertEqual([e.kind for e in out], ["content"])
        self.assertIn("{not json", out[0].text)

    def test_an_unterminated_call_is_text_at_the_end(self):
        p = self.parser()
        self.assertEqual(p.feed(OPEN + "abc" + NAME + "get_weather" + ARGS + '{"city"'), [])
        out = p.finish("length")
        self.assertEqual([e.kind for e in out], ["content"])
        self.assertIn("<|tool_call:begin|>abc", out[0].text)
        self.assertTrue(p.clean)

    def test_no_tools_declared_and_not_declared_only_accepts_a_call(self):
        p = frontend.SolarOutputParser(tools=None, declared_only=False)
        self.assertEqual([e.kind for e in p.feed(CALL)], ["tool_call"])
        q = frontend.SolarOutputParser(tools=None, declared_only=True)
        self.assertEqual([e.kind for e in q.feed(CALL)], ["content"])

    def test_the_xml_fallback_the_model_drifts_into(self):
        # measured in OpenCode: `<tool_call>shell cmd="..." workdir="..."</tool_call>` came out as text
        p = frontend.SolarOutputParser(tools=[{"name": "shell"}])
        out = p.feed('<tool_call>shell cmd="dir C:\\x" workdir="C:\\x"</tool_call>')
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.name, "shell")
        self.assertEqual(calls[0].call.arguments, {"command": "dir C:\\x", "workdir": "C:\\x"})

    def test_the_xml_function_form(self):
        p = frontend.SolarOutputParser(tools=[{"name": "shell"}])
        out = p.feed("<tool_call><function=shell><parameter=command>dir</parameter></function></tool_call>")
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual((calls[0].call.name, calls[0].call.arguments), ("shell", {"command": "dir"}))

    def test_xml_split_across_tokens(self):
        p = frontend.SolarOutputParser(tools=[{"name": "shell"}])
        out = []
        for ch in '<tool_call>shell cmd="dir"</tool_call>':
            out += p.feed(ch)
        self.assertEqual([e.kind for e in out if e.kind == "tool_call"], ["tool_call"])
        self.assertEqual("".join(e.text for e in out if e.kind == "content"), "")

    def test_an_unterminated_xml_call_is_text_at_the_end(self):
        p = frontend.SolarOutputParser(tools=[{"name": "shell"}])
        self.assertEqual(p.feed('<tool_call>shell cmd="dir"'), [])
        out = p.finish("stop")
        self.assertIn("<tool_call>shell", out[0].text)

    def test_a_call_whose_name_marker_is_missing(self):
        # measured in OpenCode: `<|tool_call:begin|>8888888888>tool<|tool_call:args|>{"command": "read", ...}`
        p = frontend.SolarOutputParser(tools=[{"name": "read"}])
        raw = ('<|tool_call:begin|>8888888888>tool<|tool_call:args|>'
               '{"command": "read", "arguments": {"path": "C:\\\\x\\\\a.docx"}}<|tool_call:end|>')
        out = p.feed(raw)
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.name, "read")
        self.assertEqual(calls[0].call.arguments, {"path": "C:\\x\\a.docx"})

    def test_the_tool_result_ids_reach_the_template(self):
        # Solar's template writes <|tool_response:begin|>ID<|tool_response:name|>NAME; the OpenAI spelling of those
        # fields has to survive openai_to_messages, or every second turn of an agent renders with an empty id/name.
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": "abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "abc123", "content": "sunny"}]
        out, _tools, _kw = frontend.openai_to_messages({"messages": msgs}, keep_ids=True)
        self.assertEqual(out[0]["tool_calls"][0]["id"], "abc123")
        self.assertEqual(out[1]["tool_call_id"], "abc123")


class ToolNudge(unittest.TestCase):
    """Solar writes a plan and ends the turn on an indirect task; the server appends an imperative for work messages."""

    def test_a_task_gets_the_imperative(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "C:\\Translate\\Novel_ForTest 내의 파일 확인 하고 정리해줘"}]
        out = server._solar_tool_nudge(msgs)
        self.assertTrue(out[1]["content"].endswith(server.SOLAR_TOOL_NUDGE))
        self.assertEqual(msgs[1]["content"], "C:\\Translate\\Novel_ForTest 내의 파일 확인 하고 정리해줘")   # a copy
        self.assertEqual(out[0], msgs[0])

    def test_a_tool_result_gets_a_continue_turn(self):
        msgs = [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "a", "content": "sub1/\nsub2/"}]
        out = server._solar_tool_nudge(msgs)
        self.assertEqual(len(out), len(msgs) + 1)
        self.assertEqual(out[-1], {"role": "user", "content": server.SOLAR_AFTER_TOOL})
        self.assertEqual(out[:-1], msgs)

    def test_a_question_is_left_alone(self):
        msgs = [{"role": "user", "content": "안녕, 오늘 기분 어때?"}]
        self.assertIs(server._solar_tool_nudge(msgs), msgs)
        other = [{"role": "user", "content": "hello there"}]
        self.assertIs(server._solar_tool_nudge(other), other)


if __name__ == "__main__":
    unittest.main()
