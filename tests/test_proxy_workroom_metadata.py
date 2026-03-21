from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.workroom_metadata import (
    workroom_max_rounds_for_definition,
    workroom_participants_for_definition,
    workroom_shape_for_definition,
    workroom_turn_sequence_for_definition,
)


class ProxyWorkroomMetadataTests(unittest.TestCase):
    def test_workroom_metadata_helpers_normalize_values(self) -> None:
        definition = DefinitionDocument(
            id="discussion-room",
            path=Path("discussion-room.md"),
            metadata={
                "id": "discussion-room",
                "shape": "discussion",
                "step_groups": [
                    [" architect ", "reviewer"],
                    ["reviewer", "brainstormer"],
                ],
                "max_rounds": 6,
            },
            body="## Purpose\nDiscussion.",
            sections={"Purpose": "Discussion."},
        )

        self.assertEqual(workroom_shape_for_definition(definition), "discussion")
        self.assertEqual(
            workroom_participants_for_definition(definition),
            ("architect", "reviewer", "brainstormer"),
        )
        self.assertEqual(workroom_max_rounds_for_definition(definition), 6)
        self.assertEqual(
            workroom_turn_sequence_for_definition(definition),
            ("architect", "reviewer", "reviewer", "brainstormer"),
        )
