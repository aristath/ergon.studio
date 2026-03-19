from __future__ import annotations

import unittest
from types import SimpleNamespace

from ergon_studio.workflow_runtime import WorkflowReviewVerdict, _format_review_summary, _has_required_tool_calls, _parse_review_verdict, _required_tool_names, _supports_auto_repair


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

    def test_auto_repair_support_is_limited_to_delivery_workflows(self) -> None:
        self.assertTrue(_supports_auto_repair("standard-build"))
        self.assertTrue(_supports_auto_repair("best-of-n"))
        self.assertTrue(_supports_auto_repair("test-driven-repair"))
        self.assertFalse(_supports_auto_repair("architecture-first"))

    def test_required_tool_names_match_execution_roles(self) -> None:
        self.assertEqual(_required_tool_names("coder"), ("write_file", "patch_file"))
        self.assertEqual(_required_tool_names("tester"), ("run_command",))
        self.assertEqual(_required_tool_names("architect"), ())

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
