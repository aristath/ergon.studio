from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.definitions import load_definition
from ergon_studio.workroom_layout import (
    referenced_agents_for_definition,
    workroom_participants_for_definition,
)


class WorkroomLayoutTests(unittest.TestCase):
    def test_participants_preserve_order_and_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "best-of-n.md"
            definition_path.write_text(
                """---
id: best-of-n
name: Best of N
participants:
  - coder
  - coder
  - coder
  - reviewer
---
## Purpose
Generate multiple candidates.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            self.assertEqual(
                workroom_participants_for_definition(definition),
                ("coder", "coder", "coder", "reviewer"),
            )
            self.assertEqual(
                referenced_agents_for_definition(definition),
                ("coder", "coder", "coder", "reviewer"),
            )

    def test_participants_require_explicit_list(self) -> None:
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
            with self.assertRaisesRegex(ValueError, "must declare `participants`"):
                workroom_participants_for_definition(definition)

    def test_participants_reject_non_list_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
participants: coder
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(ValueError, "participants must be a list"):
                workroom_participants_for_definition(definition)

    def test_participants_reject_empty_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
participants:
  - ""
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(
                ValueError,
                "participants must be non-empty strings",
            ):
                workroom_participants_for_definition(definition)
