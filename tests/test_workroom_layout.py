from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.definitions import load_definition
from ergon_studio.workroom_layout import (
    discussion_turns_for_definition,
    staged_groups_for_definition,
    workroom_kind_for_definition,
    workroom_participants_for_definition,
    workroom_turn_sequence_for_definition,
)


class WorkroomLayoutTests(unittest.TestCase):
    def test_staged_groups_support_parallel_stage_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "best-of-n.md"
            definition_path.write_text(
                """---
id: best-of-n
name: Best of N
stages:
  - [coder, coder, coder]
  - [reviewer]
---
## Purpose
Generate multiple candidates in parallel.
""",
                encoding="utf-8",
            )

            stage_groups = staged_groups_for_definition(
                load_definition(definition_path)
            )

            self.assertEqual(
                stage_groups,
                (("coder", "coder", "coder"), ("reviewer",)),
            )
            self.assertEqual(
                workroom_participants_for_definition(load_definition(definition_path)),
                ("coder", "reviewer"),
            )

    def test_staged_groups_reject_invalid_stage_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
stages:
  - []
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(
                ValueError,
                "stages must contain non-empty groups",
            ):
                staged_groups_for_definition(definition)

    def test_staged_groups_require_explicit_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(ValueError, "must declare `stages`"):
                staged_groups_for_definition(definition)

    def test_discussion_turns_preserve_order_and_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "debate.md"
            definition_path.write_text(
                """---
id: debate
name: Debate
turns:
  - architect
  - brainstormer
  - architect
  - reviewer
---
## Purpose
Compare competing approaches.
""",
                encoding="utf-8",
            )

            turns = discussion_turns_for_definition(
                load_definition(definition_path)
            )

            self.assertEqual(
                turns,
                ("architect", "brainstormer", "architect", "reviewer"),
            )
            definition = load_definition(definition_path)
            self.assertEqual(workroom_kind_for_definition(definition), "discussion")
            self.assertEqual(
                workroom_turn_sequence_for_definition(definition),
                ("architect", "brainstormer", "architect", "reviewer"),
            )

    def test_workroom_kind_rejects_missing_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(
                ValueError,
                "must declare either `stages` or `turns`",
            ):
                workroom_kind_for_definition(definition)

    def test_workroom_kind_rejects_ambiguous_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
stages:
  - coder
turns:
  - reviewer
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(
                ValueError,
                "cannot declare both `stages` and `turns`",
            ):
                workroom_kind_for_definition(definition)
