from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.grouped_workflow_executor import ProxyGroupedWorkflowExecutor
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.selection_outcome import ProxySelectionOutcome
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)


class GroupedWorkflowExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_steps_and_emits_summary(self) -> None:
        streamed_agents: list[str] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Plan", "coder": "Built"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Build it",
                state=state,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "coder"])
        self.assertEqual(
            summary_calls,
            [("Built", ("architect: Plan", "coder: Built"))],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[2], ProxyContentDeltaEvent)

    async def test_execute_labels_parallel_role_instances(self) -> None:
        streamed_prompts: list[str] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_best_of_n_definition(),
                goal="Try a few approaches",
                state=state,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("coder[1]: Idea A", reasoning)
        self.assertIn("coder[2]: Idea B", reasoning)
        self.assertIn("coder[3]: Chosen", reasoning)
        self.assertEqual(
            summary_calls,
            [
                (
                    "coder[1]: Idea A\ncoder[2]: Idea B\ncoder[3]: Chosen",
                    ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
                )
            ],
        )
        self.assertIn("Current staffed instance: coder[1]", streamed_prompts[0])
        self.assertIn("instance 2 of 3", streamed_prompts[1])

    async def test_execute_expands_grouped_role_counts_from_staffing_plan(
        self,
    ) -> None:
        summary_calls: list[tuple[str, tuple[str, ...]]] = []
        coder_outputs = iter(["Idea A", "Idea B", "Idea C"])

        async def _stream_text_agent(**_kwargs):
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Try a few approaches",
                specialists=("coder",),
                specialist_counts=(("coder", 3),),
                state=state,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertNotIn("architect:", reasoning)
        self.assertIn("coder[1]: Idea A", reasoning)
        self.assertIn("coder[2]: Idea B", reasoning)
        self.assertIn("coder[3]: Idea C", reasoning)
        self.assertEqual(
            summary_calls,
            [
                (
                    "coder[1]: Idea A\ncoder[2]: Idea B\ncoder[3]: Idea C",
                    ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Idea C"),
                )
            ],
        )

    async def test_execute_runs_parallel_attempt_group_concurrently(self) -> None:
        summary_calls: list[tuple[str, tuple[str, ...]]] = []
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**_kwargs):
            await asyncio.sleep(0.05)
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()

        started_at = time.perf_counter()
        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_definition(),
            goal="Try a few approaches",
            state=state,
        )]
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.12)
        self.assertEqual(
            summary_calls,
            [
                (
                    "coder[1]: Idea A\ncoder[2]: Idea B\ncoder[3]: Chosen",
                    ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
                )
            ],
        )

    async def test_execute_falls_back_to_sequential_when_parallel_attempt_uses_tools(
        self,
    ) -> None:
        call_count = 0
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                kwargs["final_response_sink"](
                    _fake_response_with_tool_call(kwargs["agent_id"], call_count)
                )
                yield f"Draft {call_count}"
                return
            yield f"Final {call_count}"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
            tools=(_host_tool("read_file"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_best_of_n_definition(),
                goal="Try a few approaches",
                state=state,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("rerunning this staffed group sequentially", reasoning)
        self.assertEqual(call_count, 6)
        self.assertEqual(
            summary_calls,
            [
                (
                    "coder[1]: Final 4\ncoder[2]: Final 5\ncoder[3]: Final 6",
                    ("coder[1]: Final 4", "coder[2]: Final 5", "coder[3]: Final 6"),
                )
            ],
        )

    async def test_next_stage_receives_parallel_attempts_as_comparison_candidates(
        self,
    ) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Selected best option"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_review_definition(),
            goal="Pick the best one",
            state=state,
            continuation=ContinuationState(
                mode="workflow",
                workflow_id="best-of-n",
                workflow_specialists=("coder", "reviewer"),
                last_stage_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
                last_stage_parallel_attempts=True,
                step_index=1,
                agent_index=0,
                agent_id="reviewer",
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                workflow_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
            ),
        )]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Alternative attempts from the previous stage", reviewer_prompt)
        self.assertIn("coder[1]: Idea A", reviewer_prompt)
        self.assertIn("coder[2]: Idea B", reviewer_prompt)
        self.assertIn("Treat these as competing options", reviewer_prompt)

    async def test_next_stage_receives_explicit_comparison_directive(self) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Selected best option"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_review_definition(),
            goal="Pick the best one",
            state=state,
            loop_state=ProxyDecisionLoopState(
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                current_comparison_mode="select_best",
                current_comparison_criteria="Prefer the safer and simpler approach.",
            ),
            continuation=ContinuationState(
                mode="workflow",
                workflow_id="best-of-n",
                workflow_specialists=("coder", "reviewer"),
                last_stage_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
                last_stage_parallel_attempts=True,
                step_index=1,
                agent_index=0,
                agent_id="reviewer",
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                workflow_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
            ),
        )]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Current comparison task:", reviewer_prompt)
        self.assertIn("Choose the strongest candidate", reviewer_prompt)
        self.assertIn("Comparison criteria:", reviewer_prompt)
        self.assertIn("Prefer the safer and simpler approach.", reviewer_prompt)

    async def test_stage_prompt_receives_playbook_round_assignment_and_focus(
        self,
    ) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Reviewer chose candidate 2"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_review_definition(),
            goal="Pick the best one",
            workflow_request="Compare the two alternatives and choose one winner.",
            workflow_focus="compare",
            state=state,
            continuation=ContinuationState(
                mode="workflow",
                workflow_id="best-of-n",
                workflow_specialists=("coder", "reviewer"),
                last_stage_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
                last_stage_parallel_attempts=True,
                step_index=1,
                agent_index=0,
                agent_id="reviewer",
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                workflow_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
            ),
        )]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Current playbook round assignment:", reviewer_prompt)
        self.assertIn(
            "Compare the two alternatives and choose one winner.",
            reviewer_prompt,
        )
        self.assertIn("Current playbook round focus:", reviewer_prompt)
        self.assertIn("compare", reviewer_prompt)
        self.assertIn("Judge alternatives", reviewer_prompt)

    async def test_comparison_stage_emits_structured_selection_outcome(
        self,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        async def _stream_text_agent(**_kwargs):
            yield "Choose coder[2] and polish it"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            yield ProxyContentDeltaEvent("Final summary")

        async def _select_comparison_outcome(**kwargs):
            captured_kwargs.update(kwargs)
            return ProxySelectionOutcome(
                mode="select_best",
                selected_candidate_index=1,
                selected_candidate_text="coder[2]: Idea B",
                summary="Candidate 2 is clearer and safer.",
                next_refinement="Polish candidate 2 into the final implementation.",
            )

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        async for _event in executor.execute(
            request=request,
            definition=_review_then_polish_definition(),
            goal="Pick the best one",
            state=state,
            loop_state=ProxyDecisionLoopState(
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                current_comparison_mode="select_best",
                current_comparison_criteria="Prefer the safer and simpler approach.",
            ),
            continuation=ContinuationState(
                mode="workflow",
                workflow_id="best-of-n",
                workflow_specialists=("coder", "reviewer"),
                last_stage_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
                last_stage_parallel_attempts=True,
                step_index=1,
                agent_index=0,
                agent_id="reviewer",
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                workflow_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
            ),
            result_sink=lambda result: captured_kwargs.setdefault("result", result),
        ):
            pass

        move_result = captured_kwargs["result"]
        self.assertIsInstance(move_result, ProxyMoveResult)
        assert isinstance(move_result, ProxyMoveResult)
        self.assertEqual(
            captured_kwargs["comparison_candidates"],
            ("coder[1]: Idea A", "coder[2]: Idea B"),
        )
        self.assertEqual(captured_kwargs["comparison_mode"], "select_best")
        self.assertIsNotNone(move_result.selection_outcome)
        assert move_result.selection_outcome is not None
        self.assertEqual(
            move_result.current_brief,
            "Candidate 2 is clearer and safer.",
        )
        self.assertEqual(
            move_result.selection_outcome.selected_candidate_text,
            "coder[2]: Idea B",
        )
        self.assertTrue(move_result.selection_outcome_changed)
        self.assertIn(
            "Orchestrator comparison result (select_best)",
            move_result.worklog_lines[-1],
        )
        self.assertIn("selected coder[2]: Idea B", move_result.worklog_lines[-1])
        self.assertIn(
            "Polish candidate 2 into the final implementation.",
            move_result.worklog_lines[-1],
        )
        self.assertIsNotNone(move_result.workflow_progress)
        assert move_result.workflow_progress is not None
        self.assertEqual(
            move_result.workflow_progress.selection_outcome,
            move_result.selection_outcome,
        )

    async def test_grouped_workflow_progress_preserves_playbook_focus(self) -> None:
        captured: dict[str, ProxyMoveResult] = {}

        async def _stream_text_agent(**_kwargs):
            yield "Plan"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**_kwargs):
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        async for _event in executor.execute(
            request=request,
            definition=_definition(),
            goal="Build it",
            workflow_focus="plan",
            state=state,
            result_sink=lambda result: captured.setdefault("result", result),
        ):
            pass

        move_result = captured["result"]
        self.assertIsNotNone(move_result.workflow_progress)
        assert move_result.workflow_progress is not None
        self.assertEqual(move_result.workflow_progress.workflow_focus, "plan")


    async def test_next_stage_receives_structured_selection_outcome(
        self,
    ) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Polished implementation"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_comparison_outcome=_select_comparison_outcome,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Polish the winner"),),
        )
        state = ProxyTurnState()

        [event async for event in executor.execute(
            request=request,
            definition=_review_then_polish_definition(),
            goal="Polish the winner",
            state=state,
            continuation=ContinuationState(
                mode="workflow",
                workflow_id="best-of-n",
                workflow_specialists=("reviewer", "coder"),
                selection_outcome=ProxySelectionOutcome(
                    mode="select_best",
                    selected_candidate_index=1,
                    selected_candidate_text="coder[2]: Idea B",
                    summary="Candidate 2 is clearer and safer.",
                    next_refinement="Polish candidate 2 into the final implementation.",
                ),
                step_index=2,
                agent_index=0,
                agent_id="coder",
                goal="Polish the winner",
                current_brief="Reviewer picked candidate 2.",
                workflow_outputs=(
                    "coder[1]: Idea A",
                    "coder[2]: Idea B",
                    "reviewer: Candidate 2 wins",
                ),
            ),
        )]

        coder_prompt = streamed_prompts[0]
        self.assertIn("Latest structured comparison outcome:", coder_prompt)
        self.assertIn("Selected candidate:", coder_prompt)
        self.assertIn("coder[2]: Idea B", coder_prompt)
        self.assertIn("Suggested refinement:", coder_prompt)
        self.assertIn(
            "Polish candidate 2 into the final implementation.",
            coder_prompt,
        )


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="standard-build",
        path=Path("standard-build.md"),
        metadata={
            "id": "standard-build",
            "orchestration": "sequential",
            "steps": ["architect", "coder"],
        },
        body="## Purpose\nBuild.",
        sections={"Purpose": "Build."},
    )


def _best_of_n_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="best-of-n",
        path=Path("best-of-n.md"),
        metadata={
            "id": "best-of-n",
            "orchestration": "grouped",
            "step_groups": [["coder", "coder", "coder"]],
        },
        body="## Purpose\nCompare attempts.",
        sections={"Purpose": "Compare attempts."},
    )


def _best_of_n_review_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="best-of-n",
        path=Path("best-of-n.md"),
        metadata={
            "id": "best-of-n",
            "orchestration": "grouped",
            "step_groups": [["coder", "coder"], ["reviewer"]],
        },
        body="## Purpose\nCompare attempts.",
        sections={"Purpose": "Compare attempts."},
    )


def _review_then_polish_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="best-of-n",
        path=Path("best-of-n.md"),
        metadata={
            "id": "best-of-n",
            "orchestration": "grouped",
            "step_groups": [["coder", "coder"], ["reviewer"], ["coder"]],
        },
        body="## Purpose\nCompare, select, then polish.",
        sections={"Purpose": "Compare, select, then polish."},
    )


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"Tool {name}",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )


def _fake_response_with_tool_call(agent_id: str, call_index: int):
    class _FakeContent:
        def __init__(self) -> None:
            self.type = "function_call"
            self.call_id = f"call_{call_index}"
            self.name = "read_file"
            self.arguments = '{"path":"main.py"}'

    class _FakeMessage:
        def __init__(self) -> None:
            self.contents = [_FakeContent()]
            self.author_name = agent_id

    class _FakeResponse:
        def __init__(self) -> None:
            self.messages = [_FakeMessage()]

    return _FakeResponse()


async def _select_comparison_outcome(**_kwargs):
    return None
