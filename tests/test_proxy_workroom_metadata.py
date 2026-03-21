from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.workroom_metadata import (
    workroom_finalizers_for_definition,
    workroom_handoffs_for_definition,
    workroom_max_rounds_for_definition,
    workroom_participants_for_definition,
    workroom_selection_sequence_for_definition,
    workroom_shape_for_definition,
    workroom_start_agent_for_definition,
)


class ProxyWorkroomMetadataTests(unittest.TestCase):
    def test_workroom_metadata_helpers_normalize_values(self) -> None:
        definition = DefinitionDocument(
            id="specialist-handoff",
            path=Path("specialist-handoff.md"),
            metadata={
                "id": "specialist-handoff",
                "shape": "handoff",
                "step_groups": [
                    [" architect ", "reviewer"],
                    ["reviewer", "brainstormer"],
                ],
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

        self.assertEqual(workroom_shape_for_definition(definition), "handoff")
        self.assertEqual(
            workroom_participants_for_definition(definition),
            ("architect", "reviewer", "brainstormer"),
        )
        self.assertEqual(workroom_max_rounds_for_definition(definition), 6)
        self.assertEqual(
            workroom_selection_sequence_for_definition(definition),
            ("architect", "reviewer"),
        )
        self.assertEqual(workroom_start_agent_for_definition(definition), "architect")
        self.assertEqual(workroom_finalizers_for_definition(definition), ("reviewer",))
        self.assertEqual(
            workroom_handoffs_for_definition(definition),
            {"architect": ("reviewer", "brainstormer")},
        )
