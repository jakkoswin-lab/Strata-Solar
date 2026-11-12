"""Solar-Open-100B through the server: its effort words, the template's globals and its message markers.

    python -m unittest serve.test_solar_family -v

No GPU, no pack, no model: the family's rules are pure functions and one unbound `Service.set_family`.
"""
from __future__ import annotations

import sys
import types
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import frontend                      # noqa: E402
from serve import server                        # noqa: E402

SOLAR_IDS = {"<|begin|>": 20, "<|end|>": 21, "<|think|>": 22, "<|content|>": 23, "<|flush|>": 24,
             "<|calls|>": 25, "<|tool_calls|>": 30, "<|tool_call:begin|>": 31, "<|tool_call:end|>": 32,
             "<|tool_call:name|>": 33, "<|tool_call:args|>": 34}
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
        self.assertEqual(svc.stop_ids, {2, 20, 21, 24, 25})   # eos, <|begin|>, <|end|>, <|flush|>, <|calls|>
        self.assertEqual(svc.skip_ids, {22, 23, 24, 30})      # <|think|>, <|content|>, <|flush|>, <|tool_calls|>
        self.assertTrue(svc.tmpl_ids_ok)                      # the mask's hardcoded ids match this tokenizer
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

    def test_the_bracket_form_with_a_fenced_payload(self):
        # measured in OpenCode: `<tool_call>[execute]</tool_call>` and the javascript in a ```js block after it
        p = frontend.SolarOutputParser(tools=[{"name": "execute",
                                               "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}])
        out = p.feed('<tool_call>[execute]</tool_call>\n\n```js\nconst a = 1;\n```')
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0].call.name, calls[0].call.arguments), ("execute", {"code": "const a = 1;\n"}))

    def test_the_bracket_form_fills_the_tool_s_own_field(self):
        tools = [{"name": "shell", "parameters": {"type": "object",
                                                 "properties": {"command": {"type": "string"},
                                                                "workdir": {"type": "string"}}}}]
        p = frontend.SolarOutputParser(tools=tools)
        out = p.feed('<tool_call>[shell]</tool_call>\n```powershell\ndir\n```')
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual((calls[0].call.name, calls[0].call.arguments), ("shell", {"command": "dir\n"}))

    def test_a_bracket_call_with_no_fence_is_text(self):
        p = frontend.SolarOutputParser(tools=[{"name": "execute",
                                               "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}])
        out = p.feed('<tool_call>[execute]</tool_call>\n\n그리고 계속 설명합니다.')
        self.assertEqual({e.kind for e in out}, {"content"})
        self.assertIn("그리고 계속 설명합니다", "".join(e.text for e in out))

    def test_a_bracket_call_split_into_single_characters(self):
        p = frontend.SolarOutputParser(tools=[{"name": "execute",
                                               "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}])
        out = []
        for ch in '<tool_call>[execute]</tool_call>\n```js\nlet x = 2;\n```':
            out += p.feed(ch)
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.arguments, {"code": "let x = 2;\n"})
        self.assertEqual("".join(e.text for e in out if e.kind == "content"), "")

    def test_a_shell_call_with_the_command_as_its_body(self):
        # measured: `<|tool_call:args|>cd C:\\x && python ...` - not JSON at all
        p = frontend.SolarOutputParser(tools=[{"name": "shell"}])
        raw = ('<|tool_call:begin|>8888888888<|tool_call:name|>shell<|tool_call:args|>'
               'cd C:\\x && python -m foo --mode "proofread"<|tool_call:end|><|calls|>')
        out = p.feed(raw)
        calls = [e for e in out if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.arguments, {"command": 'cd C:\\x && python -m foo --mode "proofread"'})

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

    def test_an_empty_xlsx_read_says_how_to_read_it(self):
        # measured: .xlsx is a zip too, and read returns nothing for it
        msgs = [{"role": "user", "content": "설정집 확인해줘"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "x1", "type": "function",
                     "function": {"name": "read", "arguments": json.dumps({"path": "C:\\y\\a.xlsx"})}}]},
                {"role": "tool", "tool_call_id": "x1", "content": ""}]
        out = server._solar_tool_nudge(msgs, r"C:\tools\docx_text.py", r"C:\tools\xlsx_text.py")
        self.assertIn("xlsx_text.py", out[-1]["content"])
        self.assertIn("shell 도구로", out[-1]["content"])

    def test_a_listing_that_names_a_sheet_points_at_the_reader(self):
        msgs = [{"role": "user", "content": "확인해줘"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "x2", "type": "function",
                     "function": {"name": "read", "arguments": json.dumps({"path": "C:\\y"})}}]},
                {"role": "tool", "tool_call_id": "x2", "content": "설정집_2026.xlsx\n가이드.docx\n"}]
        out = server._solar_tool_nudge(msgs, r"C:\tools\docx_text.py", r"C:\tools\xlsx_text.py")
        self.assertIn("xlsx_text.py", out[-1]["content"])

    def test_a_docx_task_is_told_how_to_read_it_first(self):
        # measured: pointed at a .docx, the model's first move is `read`, which returns nothing for a zip
        msgs = [{"role": "user", "content": r"C:\x\a.docx 내용을 확인해줘"}]
        out = server._solar_tool_nudge(msgs, r"C:\tools\docx_text.py")
        self.assertIn("docx_text.py", out[-1]["content"])
        self.assertIn("shell 도구로", out[-1]["content"])

    def test_the_settings_and_guide_the_user_named_are_listed_as_unread(self):
        # measured: the novel was read and the model answered at once, never opening 설정집/가이드
        msgs = [{"role": "user", "content": "C:\\Translate\\Novel_ForTest\\2.MT 본 안의 소설을 검수해줘\n"
                                            "가이드:\nC:\\Translate\\Novel_ForTest\\99.가이드\n"
                                            "설정집:\nC:\\Translate\\Novel_ForTest\\98.설정집"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "shell", "arguments": json.dumps(
                         {"command": "python tools\\docx_text.py \"C:\\Translate\\Novel_ForTest\\2.*\\*.docx\""})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "1: §§§第24章\n2: #제24화\n"}]
        unread = server._user_paths(msgs)
        self.assertEqual(unread, [r"C:\Translate\Novel_ForTest\99.가이드",
                                  r"C:\Translate\Novel_ForTest\98.설정집"])
        out = server._solar_tool_nudge(msgs)
        self.assertIn("아직 읽지 않은 입력", out[-1]["content"])
        self.assertIn("98.설정집", out[-1]["content"])
        self.assertIn("99.가이드", out[-1]["content"])

    def test_a_path_already_read_is_not_listed(self):
        msgs = [{"role": "user", "content": "C:\\a\\설정집 확인해줘\nC:\\a\\설정집"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "shell", "arguments": json.dumps(
                         {"command": "python tools\\xlsx_text.py \"C:\\a\\설정집\\*.xlsx\""})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sheets: 1"}]
        out = server._solar_tool_nudge(msgs)
        self.assertNotIn("아직 읽지 않은 입력", out[-1]["content"])

    def test_a_tool_result_gets_a_continue_turn(self):
        msgs = [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "a", "content": "sub1/\nsub2/"}]
        out = server._solar_tool_nudge(msgs)
        self.assertEqual(len(out), len(msgs) + 1)
        self.assertEqual(out[-1], {"role": "user", "content": server.SOLAR_AFTER_TOOL})
        self.assertEqual(out[:-1], msgs)

    def test_a_failed_tool_result_says_to_change_the_command(self):
        msgs = [{"role": "user", "content": "x"},
                {"role": "tool", "tool_call_id": "a",
                 "content": "Get-ChildItem : A positional parameter cannot be found\nExited with code 1"}]
        out = server._solar_tool_nudge(msgs)
        self.assertEqual(out[-1], {"role": "user", "content": server.SOLAR_AFTER_TOOL_ERROR})

    def test_an_empty_docx_read_says_how_to_read_it(self):
        # measured: .docx is a zip, OpenCode's read returns nothing, and Solar re-ran read three times
        msgs = [{"role": "user", "content": "확인해줘"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "read", "arguments": json.dumps({"path": "C:\\\\x\\\\a.docx"})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": ""}]
        out = server._solar_tool_nudge(msgs)
        self.assertTrue(out[-1]["content"].startswith("방금 read는"))
        self.assertIn("docx_text.py", out[-1]["content"])
        self.assertIn("shell 도구로", out[-1]["content"])

    def test_an_empty_result_of_another_tool_says_it_plainly(self):
        msgs = [{"role": "user", "content": "확인해줘"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c2", "type": "function",
                     "function": {"name": "grep", "arguments": json.dumps({"pattern": "x"})}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "   "}]
        self.assertEqual(server._solar_tool_nudge(msgs)[-1],
                         {"role": "user", "content": server.SOLAR_EMPTY_TOOL})

    def test_a_compaction_request_gets_the_fill_it_in_nudge(self):
        ask = "You MUST use this format for your response.\n<template>\n## Objective\n- [...]\n## Work State\n- [...]\n</template>"
        msgs = [{"role": "user", "content": ask}]
        out = server._solar_tool_nudge(msgs)
        self.assertTrue(out[-1]["content"].startswith(ask))
        self.assertTrue(out[-1]["content"].endswith(server.SOLAR_SUMMARY_NUDGE))

    def test_a_question_is_left_alone(self):
        msgs = [{"role": "user", "content": "안녕, 오늘 기분 어때?"}]
        self.assertIs(server._solar_tool_nudge(msgs), msgs)
        other = [{"role": "user", "content": "hello there"}]
        self.assertIs(server._solar_tool_nudge(other), other)


class ToolArguments(unittest.TestCase):
    """Measured: Solar writes Windows paths raw inside a call's JSON, so the call used to fall back to text."""

    def test_a_path_with_single_backslashes(self):
        args = frontend.json_args('{"path": "C:\\Translate\\Novel_1\\a.docx"}')
        self.assertEqual(args, {"path": r"C:\Translate\Novel_1\a.docx"})

    def test_a_letter_after_a_valid_escape_is_a_separator(self):
        # "\\t" is a legal escape (a tab), but C:\tools means the folder
        args = frontend.json_args('{"command": "python C:\\tools\\x.py"}')
        self.assertEqual(args, {"command": r"python C:\tools\x.py"})
        self.assertNotIn("\t", args["command"])

    def test_a_real_escape_is_left_alone(self):
        args = frontend.json_args('{"command": "echo a\\nb\\t"}')
        self.assertEqual(args, {"command": "echo a\nb\t"})

    def test_the_whole_call_is_read(self):
        p = frontend.SolarOutputParser(tools=[{"name": "shell"}])
        raw = ('<|tool_call:begin|>82<|tool_call:name|>shell<|tool_call:args|>'
               '{"command": "python C:\\Translate\\x.py \\"C:\\\\a\\\\b.txt\\" --start 1"}'
               '<|tool_call:end|>')
        calls = [e for e in p.feed(raw) if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].call.arguments,
                         {"command": 'python C:\\Translate\\x.py "C:\\a\\b.txt" --start 1'})


class DropSystemPrompt(unittest.TestCase):
    """The config's "drop_system_prompt": Solar's template writes the tool block itself, and an agent's system prompt
    (OpenCode's code-mode `execute`) pulls it out of its own call format."""

    def _svc(self, drop):
        class T:
            def render(self, messages, tools=None, add_generation_prompt=True, **kw):
                return json.dumps({"messages": messages, "tools": tools})
        svc = object.__new__(server.Service)
        svc.family, svc.drop_system, svc.effort_end, svc.template = "glm4moe", drop, False, T()
        svc.docx_tool = r"tools\docx_text.py"
        svc.xlsx_tool = r"tools\xlsx_text.py"
        return svc

    def test_with_tools_drops_the_system_turn(self):
        msgs = [{"role": "system", "content": "you are opencode"}, {"role": "user", "content": "파일 확인해줘"}]
        out = json.loads(self._svc("with_tools").render_prompt(msgs, [{"name": "shell", "parameters": {}}], {}))
        self.assertEqual([m["role"] for m in out["messages"]], ["user"])
        self.assertEqual(out["tools"], [{"type": "function", "function": {"name": "shell", "parameters": {}}}])

    def test_without_tools_the_system_turn_stays(self):
        msgs = [{"role": "system", "content": "번역 규칙"}, {"role": "user", "content": "번역해줘"}]
        out = json.loads(self._svc("with_tools").render_prompt(msgs, None, {}))
        self.assertEqual([m["role"] for m in out["messages"]], ["system", "user"])

    def test_true_drops_it_even_without_tools(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "안녕"}]
        out = json.loads(self._svc(True).render_prompt(msgs, None, {}))
        self.assertEqual([m["role"] for m in out["messages"]], ["user"])


if __name__ == "__main__":
    unittest.main()
