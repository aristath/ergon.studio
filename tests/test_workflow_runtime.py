from __future__ import annotations

import unittest
from types import SimpleNamespace

from ergon_studio.workflow_runtime import _ExecutionTracker, WorkflowFollowupDecision, WorkflowReviewVerdict, _format_review_summary, _has_required_tool_calls, _parse_followup_decision, _parse_review_verdict, _required_tool_names, _required_tool_names_for_workflow, _workflow_review_evidence_lines


class WorkflowRuntimeTests(unittest.TestCase):
    def test_parse_review_verdict_reads_json_object(self) -> None:
        verdict = _parse_review_verdict('{"accepted": true, "summary": "Ship it."}')

        self.assertEqual(verdict, WorkflowReviewVerdict(accepted=True, summary="Ship it."))

    def test_parse_review_verdict_reads_findings_and_replan_fields(self) -> None:
        verdict = _parse_review_verdict(
            '{"accepted": false, "summary": "Wrong direction.", "findings": ["Missing CLI output"], "requires_replan": true, "replan_summary": "Replan around a CLI-first flow."}'
        )

        self.assertEqual(
            verdict,
            WorkflowReviewVerdict(
                accepted=False,
                summary="Wrong direction.",
                findings=("Missing CLI output",),
                requires_replan=True,
                replan_summary="Replan around a CLI-first flow.",
            ),
        )

    def test_parse_review_verdict_extracts_fenced_json(self) -> None:
        verdict = _parse_review_verdict('```json\n{"accepted": false, "summary": "Missing tests."}\n```')

        self.assertEqual(verdict, WorkflowReviewVerdict(accepted=False, summary="Missing tests."))

    def test_format_review_summary_prefixes_verdict(self) -> None:
        summary = _format_review_summary(WorkflowReviewVerdict(accepted=False, summary="Implementation is incomplete."))

        self.assertEqual(summary, "REJECTED: Implementation is incomplete.")

    def test_parse_followup_decision_reads_clarification_request(self) -> None:
        decision = _parse_followup_decision(
            '{"action": "clarify", "summary": "Ask the tester for proof.", "agent_id": "tester", "request": "Run one concrete command and report the output.", "tool_mode": "default"}'
        )

        self.assertEqual(
            decision,
            WorkflowFollowupDecision(
                action="clarify",
                summary="Ask the tester for proof.",
                agent_id="tester",
                request="Run one concrete command and report the output.",
                tool_mode="default",
            ),
        )

    def test_parse_followup_decision_reads_custom_step_groups(self) -> None:
        decision = _parse_followup_decision(
            '{"action": "repair", "summary": "Use a custom repair path.", "request": "Investigate first, then patch.", "step_groups": [["tester"], ["fixer"], ["reviewer"]], "tool_mode": "default"}'
        )

        self.assertEqual(
            decision,
            WorkflowFollowupDecision(
                action="repair",
                summary="Use a custom repair path.",
                agent_id=None,
                request="Investigate first, then patch.",
                step_groups=(("tester",), ("fixer",), ("reviewer",)),
                tool_mode="default",
            ),
        )

    def test_required_tool_names_match_execution_roles(self) -> None:
        self.assertEqual(_required_tool_names("coder"), ("write_file", "patch_file"))
        self.assertEqual(_required_tool_names("tester"), ("run_command",))
        self.assertEqual(_required_tool_names("architect"), ())

    def test_workflow_required_tool_names_use_metadata_override(self) -> None:
        runtime = SimpleNamespace(
            registry=SimpleNamespace(
                workflow_definitions={
                    "custom": SimpleNamespace(
                        metadata={
                            "tool_evidence": {
                                "coder": ["run_command"],
                                "tester": [],
                            }
                        }
                    )
                }
            )
        )

        self.assertEqual(
            _required_tool_names_for_workflow(runtime=runtime, workflow_id="custom", agent_id="coder"),
            ("run_command",),
        )
        self.assertEqual(
            _required_tool_names_for_workflow(runtime=runtime, workflow_id="custom", agent_id="tester"),
            (),
        )
        self.assertEqual(
            _required_tool_names_for_workflow(runtime=runtime, workflow_id="custom", agent_id="reviewer"),
            ("list_files", "read_file", "search_files", "run_command"),
        )

    def test_required_tool_check_uses_completed_new_calls(self) -> None:
        tool_calls = [
            SimpleNamespace(id="old", thread_id="thread-1", status="completed", tool_name="read_file"),
            SimpleNamespace(id="new", thread_id="thread-1", status="completed", tool_name="run_command"),
            SimpleNamespace(id="other", thread_id="thread-2", status="completed", tool_name="run_command"),
        ]

        self.assertTrue(
            _has_required_tool_calls(
                tool_calls=tool_calls,
                thread_id="thread-1",
                new_tool_ids={"old"},
                required_tools=("run_command",),
            )
        )
        self.assertFalse(
            _has_required_tool_calls(
                tool_calls=tool_calls,
                thread_id="thread-1",
                new_tool_ids={"old", "new"},
                required_tools=("run_command",),
            )
        )

    def test_workflow_review_evidence_lines_include_recorded_file_and_command_evidence(self) -> None:
        runtime = SimpleNamespace(
            list_tool_calls_for_workflow_run=lambda workflow_run_id: [
                SimpleNamespace(
                    id="tool-1",
                    status="completed",
                    tool_name="write_file",
                    agent_id="coder",
                )
            ],
            read_tool_call_request=lambda tool_call_id: '{"path": "calculator.py"}',
            list_command_runs_for_workflow_run=lambda workflow_run_id: [
                SimpleNamespace(
                    id="cmd-1",
                    status="completed",
                    command="python3 calculator.py 2 + 2",
                    exit_code=0,
                )
            ],
            command_store=SimpleNamespace(
                read_command_output=lambda command_run: "4\n",
            ),
        )

        lines = _workflow_review_evidence_lines(runtime, "workflow-run-1")

        self.assertEqual(
            lines,
            [
                "Recorded file changes:",
                "- write_file by coder: calculator.py",
                "",
                "Recorded command runs:",
                "- python3 calculator.py 2 + 2 -> exit 0; output: 4",
                "",
            ],
        )
