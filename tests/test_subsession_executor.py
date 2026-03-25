from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.models import ProxyReasoningDeltaEvent, ProxyToolCall
from ergon_studio.proxy.session_overlay import SessionOverlay
from ergon_studio.proxy.subsession_executor import SubSessionExecutor
from ergon_studio.response_stream import ResponseStream


class SubSessionExecutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _overlay(self) -> SessionOverlay:
        return SessionOverlay(root=self.tmp / "overlay")

    # --- basic ---

    async def test_agent_with_no_tool_calls_returns_text(self) -> None:
        executor = SubSessionExecutor(
            stream_text_agent=_fake_stream_text_agent(
                [AgentRunResult(text="Final answer", tool_calls=())]
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Do the thing",
            session_id="s1",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        result = await stream.get_final_response()
        self.assertEqual(result, "Final answer")

    async def test_final_result_is_last_agent_text(self) -> None:
        executor = SubSessionExecutor(
            stream_text_agent=_fake_stream_text_agent(
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="read_file",
                                arguments_json=f'{{"path": "{self.tmp}/x.py"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="All done", tool_calls=()),
                ]
            )
        )
        overlay = self._overlay()
        (self.tmp / "x.py").write_text("content", encoding="utf-8")
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s2",
            overlay=overlay,
            model_id="test-model",
        )
        async for _ in stream:
            pass
        result = await stream.get_final_response()
        self.assertEqual(result, "All done")

    # --- workspace tool handling ---

    async def test_agent_read_file_tool_call_handled_internally(self) -> None:
        real_file = self.tmp / "foo.py"
        real_file.write_text("real content", encoding="utf-8")

        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="read_file",
                                arguments_json=f'{{"path": "{real_file}"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Done reading", tool_calls=()),
                ],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Read the file",
            session_id="s3",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        result = await stream.get_final_response()

        self.assertEqual(result, "Done reading")
        # agent was invoked twice
        self.assertEqual(len(invocations), 2)
        # second invocation received the tool result as a conversation message
        second_conv = invocations[1]["conversation_messages"]
        tool_msg = next(
            (m for m in second_conv if m.role == "tool"), None
        )
        self.assertIsNotNone(tool_msg)
        self.assertEqual(tool_msg.tool_call_id, "tc1")
        self.assertIn("real content", tool_msg.content)

    async def test_agent_read_file_overlay_takes_precedence_over_real(self) -> None:
        real_file = self.tmp / "foo.py"
        real_file.write_text("real content", encoding="utf-8")
        overlay = self._overlay()
        overlay.write_file(str(real_file), "overlay content")

        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="read_file",
                                arguments_json=f'{{"path": "{real_file}"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Done", tool_calls=()),
                ],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s4",
            overlay=overlay,
            model_id="test-model",
        )
        async for _ in stream:
            pass
        second_conv = invocations[1]["conversation_messages"]
        tool_msg = next(m for m in second_conv if m.role == "tool")
        self.assertIn("overlay content", tool_msg.content)

    async def test_agent_write_file_tool_call_writes_to_overlay(self) -> None:
        target = self.tmp / "out.py"
        overlay = self._overlay()
        executor = SubSessionExecutor(
            stream_text_agent=_fake_stream_text_agent(
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="write_file",
                                arguments_json=f'{{"path": "{target}", "content": "written"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Done writing", tool_calls=()),
                ]
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Write a file",
            session_id="s5",
            overlay=overlay,
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        # real file untouched
        self.assertFalse(target.exists())
        # overlay has the content
        self.assertEqual(overlay.read_file(str(target)), "written")

    async def test_agent_list_files_tool_call_handled_internally(self) -> None:
        dir_path = self.tmp / "src"
        dir_path.mkdir()
        (dir_path / "a.py").write_text("", encoding="utf-8")

        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="list_files",
                                arguments_json=f'{{"directory": "{dir_path}"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Listed", tool_calls=()),
                ],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="List files",
            session_id="s6",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        second_conv = invocations[1]["conversation_messages"]
        tool_msg = next(m for m in second_conv if m.role == "tool")
        self.assertIn("a.py", tool_msg.content)

    async def test_read_file_not_found_returns_error_message(self) -> None:
        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="read_file",
                                arguments_json='{"path": "/absolutely/nonexistent/file.py"}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Handled", tool_calls=()),
                ],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s7",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        second_conv = invocations[1]["conversation_messages"]
        tool_msg = next(m for m in second_conv if m.role == "tool")
        self.assertIn("not found", tool_msg.content.lower())

    async def test_multiple_workspace_tool_calls_handled_in_sequence(self) -> None:
        real_file = self.tmp / "in.py"
        real_file.write_text("source", encoding="utf-8")
        target = self.tmp / "out.py"
        overlay = self._overlay()

        executor = SubSessionExecutor(
            stream_text_agent=_fake_stream_text_agent(
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="read_file",
                                arguments_json=f'{{"path": "{real_file}"}}',
                            ),
                            ProxyToolCall(
                                id="tc2",
                                name="write_file",
                                arguments_json=f'{{"path": "{target}", "content": "done"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Complete", tool_calls=()),
                ]
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s8",
            overlay=overlay,
            model_id="test-model",
        )
        async for _ in stream:
            pass
        result = await stream.get_final_response()
        self.assertEqual(result, "Complete")
        self.assertEqual(overlay.read_file(str(target)), "done")

    # --- isolation ---

    async def test_sub_session_passes_no_host_tools(self) -> None:
        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [AgentRunResult(text="Done", tool_calls=())],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s9",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        host_tools = invocations[0]["host_tools"]
        self.assertEqual(tuple(host_tools), ())

    async def test_sub_session_extra_tools_are_only_workspace_tools(self) -> None:
        from ergon_studio.proxy.workspace_tools import WORKSPACE_TOOL_NAMES

        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [AgentRunResult(text="Done", tool_calls=())],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s10",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        extra_tool_names = {t.name for t in invocations[0]["extra_tools"]}
        self.assertEqual(extra_tool_names, WORKSPACE_TOOL_NAMES)

    # --- prompt framing ---

    async def test_sub_session_uses_system_prompt_framing(self) -> None:
        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [AgentRunResult(text="Done", tool_calls=())],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="My specific task",
            session_id="s-framing",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        # prompt must be the framing, not the task text
        self.assertEqual(invocations[0]["prompt_role"], "system")
        self.assertNotIn("My specific task", invocations[0]["prompt"])

    async def test_sub_session_task_injected_as_first_user_message(self) -> None:
        invocations: list[dict[str, Any]] = []
        executor = SubSessionExecutor(
            stream_text_agent=_capturing_stream_text_agent(
                invocations,
                [AgentRunResult(text="Done", tool_calls=())],
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="My specific task",
            session_id="s-task",
            overlay=self._overlay(),
            model_id="test-model",
        )
        async for _ in stream:
            pass
        await stream.get_final_response()
        conv = invocations[0]["conversation_messages"]
        self.assertGreater(len(conv), 0)
        self.assertEqual(conv[0].role, "user")
        self.assertEqual(conv[0].content, "My specific task")

    # --- loop limit ---

    async def test_loop_limit_raises_after_max_iterations(self) -> None:
        def _always_tool(**kwargs: Any) -> ResponseStream[str, AgentRunResult]:
            async def _events():  # type: ignore[return]
                return
                yield  # noqa: unreachable

            return ResponseStream(
                _events(),
                finalizer=lambda: AgentRunResult(
                    text="",
                    tool_calls=(
                        ProxyToolCall(
                            id="tc1",
                            name="write_file",
                            arguments_json=(
                                f'{{"path": "{self.tmp}/f.py", "content": "x"}}'
                            ),
                        ),
                    ),
                ),
            )

        executor = SubSessionExecutor(stream_text_agent=_always_tool)
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s-limit",
            overlay=self._overlay(),
            model_id="test-model",
        )
        with self.assertRaises(ValueError):
            async for _ in stream:
                pass

    # --- reasoning events ---

    async def test_reasoning_events_emitted_for_agent_text(self) -> None:
        executor = SubSessionExecutor(
            stream_text_agent=_fake_stream_text_agent(
                [AgentRunResult(text="Thinking...", tool_calls=())]
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s11",
            overlay=self._overlay(),
            model_id="test-model",
        )
        events = [e async for e in stream]
        reasoning_texts = [
            e.delta for e in events if isinstance(e, ProxyReasoningDeltaEvent)
        ]
        self.assertIn("Thinking...", reasoning_texts)

    async def test_no_reasoning_event_for_empty_text(self) -> None:
        executor = SubSessionExecutor(
            stream_text_agent=_fake_stream_text_agent(
                [
                    AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="tc1",
                                name="write_file",
                                arguments_json=f'{{"path": "{self.tmp}/x.py", "content": "x"}}',
                            ),
                        ),
                    ),
                    AgentRunResult(text="Done", tool_calls=()),
                ]
            )
        )
        stream = executor.execute(
            agent_id="coder",
            task="Task",
            session_id="s12",
            overlay=self._overlay(),
            model_id="test-model",
        )
        events = [e async for e in stream]
        empty_reasoning = [
            e for e in events
            if isinstance(e, ProxyReasoningDeltaEvent) and not e.delta
        ]
        self.assertEqual(empty_reasoning, [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_stream_text_agent(
    responses: list[AgentRunResult],
) -> Any:
    counter = [0]

    def _stream(**kwargs: Any) -> ResponseStream[str, AgentRunResult]:
        idx = counter[0]
        counter[0] += 1
        result = responses[idx]

        async def _events():  # type: ignore[return]
            if result.text:
                yield result.text

        return ResponseStream(_events(), finalizer=lambda r=result: r)

    return _stream


def _capturing_stream_text_agent(
    invocations: list[dict[str, Any]],
    responses: list[AgentRunResult],
) -> Any:
    counter = [0]

    def _stream(**kwargs: Any) -> ResponseStream[str, AgentRunResult]:
        invocations.append(kwargs)
        idx = counter[0]
        counter[0] += 1
        result = responses[idx]

        async def _events():  # type: ignore[return]
            if result.text:
                yield result.text

        return ResponseStream(_events(), finalizer=lambda r=result: r)

    return _stream
