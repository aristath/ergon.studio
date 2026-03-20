from __future__ import annotations

from pathlib import Path
import unittest

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.workflow_metadata import (
    workflow_finalizers_for_definition,
    workflow_handoffs_for_definition,
    workflow_max_rounds_for_definition,
    workflow_orchestration_for_definition,
    workflow_participants_for_definition,
    workflow_selection_sequence_for_definition,
    workflow_start_agent_for_definition,
)


class ProxyWorkflowMetadataTests(unittest.TestCase):
    def test_workflow_metadata_helpers_normalize_values(self) -> None:
        definition = DefinitionDocument(
            id="specialist-handoff",
            path=Path("specialist-handoff.md"),
            metadata={
                "id": "specialist-handoff",
                "orchestration": "handoff",
                "step_groups": [[" architect ", "reviewer"], ["reviewer", "brainstormer"]],
                "max_rounds": 6,
                "selection_sequence": ["architect", "reviewer"],
                "start_agent": "architect",
                "finalizers": ["reviewer"],
                "handoffs": {
                    "architect": ["reviewer", "brainstormer", "reviewer"],
                },
            },
            body="## Purpose\nHandoff.",
            sections={"Purpose": "Handoff."},
        )

        self.assertEqual(workflow_orchestration_for_definition(definition), "handoff")
        self.assertEqual(workflow_participants_for_definition(definition), ("architect", "reviewer", "brainstormer"))
        self.assertEqual(workflow_max_rounds_for_definition(definition), 6)
        self.assertEqual(workflow_selection_sequence_for_definition(definition), ("architect", "reviewer"))
        self.assertEqual(workflow_start_agent_for_definition(definition), "architect")
        self.assertEqual(workflow_finalizers_for_definition(definition), ("reviewer",))
        self.assertEqual(workflow_handoffs_for_definition(definition), {"architect": ("reviewer", "brainstormer")})
