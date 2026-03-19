from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.definitions import load_definition
from ergon_studio.workflow_compiler import compile_workflow_definition


class WorkflowCompilerTests(unittest.TestCase):
    def test_compile_workflow_definition_supports_grouped_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "best-of-n.md"
            definition_path.write_text(
                """---
id: best-of-n
name: Best of N
orchestration: concurrent
step_groups:
  - [coder, coder, coder]
  - [reviewer]
---
## Purpose
Generate multiple candidates in parallel.
""",
                encoding="utf-8",
            )

            compiled = compile_workflow_definition(load_definition(definition_path))
            mermaid = compiled.to_mermaid()

            self.assertEqual(compiled.definition_id, "best-of-n")
            self.assertIn("coder-1-1", mermaid)
            self.assertIn("reviewer-2-1", mermaid)

    def test_compile_workflow_definition_rejects_invalid_step_groups(self) -> None:
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
                compile_workflow_definition(definition)

    def test_compile_workflow_definition_supports_group_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "debate.md"
            definition_path.write_text(
                """---
id: debate
name: Debate
orchestration: group_chat
step_groups:
  - [architect, brainstormer, reviewer]
max_rounds: 6
---
## Purpose
Compare competing approaches.
""",
                encoding="utf-8",
            )

            compiled = compile_workflow_definition(load_definition(definition_path))
            mermaid = compiled.to_mermaid()

            self.assertEqual(compiled.definition_id, "debate")
            self.assertIn("group_chat_orchestrator", mermaid)
            self.assertIn("architect", mermaid)
            self.assertIn("brainstormer", mermaid)
            self.assertIn("reviewer", mermaid)

    def test_compile_workflow_definition_supports_magentic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "dynamic-open-ended.md"
            definition_path.write_text(
                """---
id: dynamic-open-ended
name: Dynamic Open Ended
orchestration: magentic
step_groups:
  - [architect, coder, reviewer]
max_rounds: 8
---
## Purpose
Adaptive specialist delegation.
""",
                encoding="utf-8",
            )

            compiled = compile_workflow_definition(load_definition(definition_path))
            mermaid = compiled.to_mermaid()

            self.assertEqual(compiled.definition_id, "dynamic-open-ended")
            self.assertIn("magentic_manager", mermaid)
            self.assertIn("architect", mermaid)
            self.assertIn("coder", mermaid)
            self.assertIn("reviewer", mermaid)

    def test_compile_workflow_definition_supports_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definition_path = Path(temp_dir) / "specialist-handoff.md"
            definition_path.write_text(
                """---
id: specialist-handoff
name: Specialist Handoff
orchestration: handoff
step_groups:
  - [architect, researcher, reviewer]
max_rounds: 6
---
## Purpose
Decentralized specialist routing.
""",
                encoding="utf-8",
            )

            compiled = compile_workflow_definition(load_definition(definition_path))
            mermaid = compiled.to_mermaid()

            self.assertEqual(compiled.definition_id, "specialist-handoff")
            self.assertIn("handoff_router", mermaid)
            self.assertIn("architect", mermaid)
            self.assertIn("researcher", mermaid)
            self.assertIn("reviewer", mermaid)
