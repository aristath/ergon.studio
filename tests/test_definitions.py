from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.definitions import load_definition, load_definitions_from_dir, save_definition


class DefinitionLoaderTests(unittest.TestCase):
    def test_load_definition_parses_yaml_frontmatter_and_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "orchestrator.md"
            definition_path.write_text(
                """---
id: orchestrator
name: Orchestrator
role: orchestrator
temperature: 0.2
tools:
  - read_file
  - write_file
can_speak_unprompted: false
---
## Identity
Lead engineer for the AI firm.

## Responsibilities
Plan work and delegate clearly.

## Rules
Do not use keyword-triggered behavior.
""",
                encoding="utf-8",
            )

            definition = load_definition(definition_path)

            self.assertEqual(definition.id, "orchestrator")
            self.assertEqual(definition.metadata["role"], "orchestrator")
            self.assertEqual(definition.metadata["tools"], ["read_file", "write_file"])
            self.assertEqual(definition.sections["Identity"], "Lead engineer for the AI firm.")
            self.assertEqual(definition.sections["Rules"], "Do not use keyword-triggered behavior.")

    def test_load_definitions_from_dir_indexes_by_definition_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definitions_dir = Path(temp_dir)
            (definitions_dir / "coder.md").write_text(
                """---
id: coder
role: coder
---
## Identity
Implements code changes.
""",
                encoding="utf-8",
            )
            (definitions_dir / "reviewer.md").write_text(
                """---
id: reviewer
role: reviewer
---
## Identity
Reviews code critically.
""",
                encoding="utf-8",
            )

            definitions = load_definitions_from_dir(definitions_dir)

            self.assertEqual(sorted(definitions.keys()), ["coder", "reviewer"])

    def test_save_definition_round_trips_markdown_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "orchestrator.md"

            saved = save_definition(
                definition_path,
                {
                    "id": "orchestrator",
                    "role": "orchestrator",
                    "temperature": 0.2,
                },
                """## Identity
Lead engineer.

## Rules
Avoid keyword-triggered behavior.
""",
            )

            self.assertEqual(saved.id, "orchestrator")
            self.assertEqual(saved.metadata["temperature"], 0.2)
            self.assertEqual(saved.sections["Identity"], "Lead engineer.")
            self.assertEqual(saved.sections["Rules"], "Avoid keyword-triggered behavior.")
