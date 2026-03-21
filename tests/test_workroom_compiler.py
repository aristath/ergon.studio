from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.definitions import load_definition
from ergon_studio.workroom_compiler import workroom_step_groups_for_definition


class WorkroomCompilerTests(unittest.TestCase):
    def test_workroom_step_groups_support_grouped_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "best-of-n.md"
            definition_path.write_text(
                """---
id: best-of-n
name: Best of N
shape: concurrent
step_groups:
  - [coder, coder, coder]
  - [reviewer]
---
## Purpose
Generate multiple candidates in parallel.
""",
                encoding="utf-8",
            )

            step_groups = workroom_step_groups_for_definition(
                load_definition(definition_path)
            )

            self.assertEqual(step_groups, (("coder", "coder", "coder"), ("reviewer",)))

    def test_workroom_step_groups_reject_invalid_step_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
step_groups:
  - []
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(ValueError, "non-empty lists"):
                workroom_step_groups_for_definition(definition)

    def test_workroom_step_groups_require_explicit_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "broken.md"
            definition_path.write_text(
                """---
id: broken
name: Broken
shape: sequential
---
## Purpose
Broken.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)
            with self.assertRaisesRegex(
                ValueError, "must declare `steps` or `step_groups`"
            ):
                workroom_step_groups_for_definition(definition)

    def test_workroom_step_groups_preserve_orchestrated_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "debate.md"
            definition_path.write_text(
                """---
id: debate
name: Debate
shape: group_chat
step_groups:
  - [architect, brainstormer, reviewer]
---
## Purpose
Compare competing approaches.
""",
                encoding="utf-8",
            )

            step_groups = workroom_step_groups_for_definition(
                load_definition(definition_path)
            )

            self.assertEqual(step_groups, (("architect", "brainstormer", "reviewer"),))
