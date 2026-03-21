from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.workroom_metadata import (
    workroom_max_rounds_for_definition,
    workroom_participants_for_definition,
    workroom_selection_hints_for_definition,
    workroom_selection_sequence_for_definition,
    workroom_shape_for_definition,
)


class ProxyWorkroomMetadataTests(unittest.TestCase):
    def test_workroom_metadata_helpers_normalize_values(self) -> None:
        definition = DefinitionDocument(
            id="discussion-room",
            path=Path("discussion-room.md"),
            metadata={
                "id": "discussion-room",
                "shape": "group_chat",
                "step_groups": [
                    [" architect ", "reviewer"],
                    ["reviewer", "brainstormer"],
                ],
                "max_rounds": 6,
                "selection_sequence": ["architect", "reviewer"],
            },
            body="## Purpose\nDiscussion.",
            sections={"Purpose": "Discussion."},
        )

        self.assertEqual(workroom_shape_for_definition(definition), "group_chat")
        self.assertEqual(
            workroom_participants_for_definition(definition),
            ("architect", "reviewer", "brainstormer"),
        )
        self.assertEqual(workroom_max_rounds_for_definition(definition), 6)
        self.assertEqual(
            workroom_selection_sequence_for_definition(definition),
            ("architect", "reviewer"),
        )
        self.assertEqual(workroom_selection_hints_for_definition(definition), ())

    def test_workroom_selection_hints_normalize_and_deduplicate(self) -> None:
        definition = DefinitionDocument(
            id="standard-build",
            path=Path("standard-build.md"),
            metadata={
                "id": "standard-build",
                "selection_hints": [
                    "staged_delivery",
                    " staged_delivery ",
                    "",
                    "ship_it",
                ],
            },
            body="## Purpose\nBuild.",
            sections={"Purpose": "Build."},
        )

        self.assertEqual(
            workroom_selection_hints_for_definition(definition),
            ("staged_delivery", "ship_it"),
        )
