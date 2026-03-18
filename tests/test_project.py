from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from ergon_studio.project import initialize_project, load_project_identity


class ProjectIdentityTests(unittest.TestCase):
    def test_initialize_project_creates_project_file_with_only_project_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            project_uuid = UUID("12345678-1234-5678-1234-567812345678")

            identity = initialize_project(project_root, project_uuid=project_uuid)

            self.assertEqual(identity.project_uuid, project_uuid)

            project_file = project_root / ".ergon.studio" / "project.json"
            self.assertTrue(project_file.exists())
            raw = json.loads(project_file.read_text(encoding="utf-8"))
            self.assertEqual(raw, {"project_uuid": str(project_uuid)})

    def test_initialize_project_reuses_existing_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            original_uuid = UUID("12345678-1234-5678-1234-567812345678")

            first = initialize_project(project_root, project_uuid=original_uuid)
            second = initialize_project(project_root, project_uuid=UUID("87654321-4321-8765-4321-876543218765"))

            self.assertEqual(first.project_uuid, original_uuid)
            self.assertEqual(second.project_uuid, original_uuid)

    def test_load_project_identity_rejects_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            project_dir = project_root / ".ergon.studio"
            project_dir.mkdir(parents=True, exist_ok=True)
            project_file = project_dir / "project.json"
            project_file.write_text(
                json.dumps(
                    {
                        "project_uuid": "12345678-1234-5678-1234-567812345678",
                        "project_name": "should-not-be-here",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "project_uuid field"):
                load_project_identity(project_root)
