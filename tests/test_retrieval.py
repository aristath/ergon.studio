from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.memory_store import MemoryStore
from ergon_studio.retrieval import RetrievalIndex


class RetrievalIndexTests(unittest.TestCase):
    def test_retrieval_index_can_index_and_query_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            (project_root / "calc.py").write_text(
                "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
                encoding="utf-8",
            )
            (project_root / "README.md").write_text(
                "# Calculator\n\nA tiny calculator CLI.\n",
                encoding="utf-8",
            )

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            retrieval = RetrievalIndex(paths)

            indexed = retrieval.rebuild_workspace_index()
            results = retrieval.query("calculator add function", limit=3)

            self.assertGreaterEqual(indexed, 2)
            self.assertTrue(any(result.path == "calc.py" for result in results))
            self.assertTrue(any("return a + b" in result.text for result in results))

    def test_retrieval_index_uses_content_detection_instead_of_suffix_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            (project_root / "component.jsx").write_text(
                "export function SaveButton() {\n  return <button>Save draft</button>;\n}\n",
                encoding="utf-8",
            )
            (project_root / "index.php").write_text(
                "<?php\nfunction greet_user() {\n    echo 'Hello from PHP';\n}\n",
                encoding="utf-8",
            )
            (project_root / "Dockerfile").write_text(
                "FROM python:3.12-slim\nRUN echo ready\n",
                encoding="utf-8",
            )

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            retrieval = RetrievalIndex(paths)
            retrieval.rebuild_workspace_index()

            react_results = retrieval.query("save draft button component", limit=5)
            php_results = retrieval.query("hello from php greet user", limit=5)
            docker_results = retrieval.query("python slim docker image", limit=5)

            self.assertTrue(any(result.path == "component.jsx" for result in react_results))
            self.assertTrue(any(result.path == "index.php" for result in php_results))
            self.assertTrue(any(result.path == "Dockerfile" for result in docker_results))

    def test_retrieval_index_skips_binary_and_junk_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            (project_root / "main.py").write_text(
                "def main():\n    return 'workspace file'\n",
                encoding="utf-8",
            )
            (project_root / "image.bin").write_bytes(b"\x00PNG\x00binary-data")
            node_modules = project_root / "node_modules"
            node_modules.mkdir()
            (node_modules / "ignored.js").write_text(
                "console.log('ignore me');\n",
                encoding="utf-8",
            )

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            retrieval = RetrievalIndex(paths)
            retrieval.rebuild_workspace_index()

            binary_results = retrieval.query("binary data png", limit=5)
            junk_results = retrieval.query("ignore me", limit=5)
            real_results = retrieval.query("workspace file", limit=5)

            self.assertFalse(any(result.path == "image.bin" for result in binary_results))
            self.assertFalse(any(result.path == "node_modules/ignored.js" for result in junk_results))
            self.assertTrue(any(result.path == "main.py" for result in real_results))

    def test_retrieval_index_ignores_empty_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            retrieval = RetrievalIndex(paths)

            self.assertEqual(retrieval.query(""), [])

    def test_retrieval_index_updates_changed_sources_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            keep_path = project_root / "keep.py"
            change_path = project_root / "change.py"
            keep_path.write_text(
                "def keep_feature():\n    return 'keep feature'\n",
                encoding="utf-8",
            )
            change_path.write_text(
                "def old_feature():\n    return 'old behavior'\n",
                encoding="utf-8",
            )

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            retrieval = RetrievalIndex(paths)
            first_count = retrieval.ensure_workspace_index(force=True)
            first_manifest = json.loads(retrieval.manifest_path.read_text(encoding="utf-8"))
            first_keep_points = list(first_manifest["sources"]["workspace:keep.py"]["point_ids"])
            first_change_points = list(first_manifest["sources"]["workspace:change.py"]["point_ids"])

            change_path.write_text(
                "def new_feature():\n    return 'new behavior'\n",
                encoding="utf-8",
            )
            second_count = retrieval.ensure_workspace_index()
            second_manifest = json.loads(retrieval.manifest_path.read_text(encoding="utf-8"))
            second_keep_points = list(second_manifest["sources"]["workspace:keep.py"]["point_ids"])
            second_change_points = list(second_manifest["sources"]["workspace:change.py"]["point_ids"])

            self.assertEqual(first_count, second_count)
            self.assertEqual(first_keep_points, second_keep_points)
            self.assertNotEqual(first_change_points, second_change_points)
            self.assertTrue(any(result.path == "change.py" for result in retrieval.query("new behavior", limit=5)))
            self.assertFalse(any(result.path == "change.py" and "old behavior" in result.text for result in retrieval.query("old behavior", limit=5)))

    def test_retrieval_index_removes_deleted_sources_without_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            retained_path = project_root / "retained.py"
            removed_path = project_root / "removed.py"
            retained_path.write_text(
                "def retained_feature():\n    return 'still here'\n",
                encoding="utf-8",
            )
            removed_path.write_text(
                "def removed_feature():\n    return 'gone now'\n",
                encoding="utf-8",
            )

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            retrieval = RetrievalIndex(paths)
            retrieval.ensure_workspace_index(force=True)
            first_manifest = json.loads(retrieval.manifest_path.read_text(encoding="utf-8"))
            retained_points = list(first_manifest["sources"]["workspace:retained.py"]["point_ids"])

            removed_path.unlink()
            retrieval.ensure_workspace_index()
            second_manifest = json.loads(retrieval.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(retained_points, second_manifest["sources"]["workspace:retained.py"]["point_ids"])
            self.assertNotIn("workspace:removed.py", second_manifest["sources"])
            self.assertFalse(any(result.path == "removed.py" for result in retrieval.query("gone now", limit=5)))

    def test_retrieval_index_can_query_memory_facts_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            memory_store = MemoryStore(paths)
            artifact_store = ArtifactStore(paths)
            memory_store.add_fact(
                fact_id="fact-1",
                scope="project",
                kind="decision",
                content="Use cedar token signing for all session cookies.",
                source="architecture-note",
                created_at=10,
                tags=("auth", "security"),
            )
            artifact_store.create_artifact(
                session_id="session-main",
                artifact_id="artifact-1",
                kind="workflow-report",
                title="Signup Flow Notes",
                content="The signup flow must validate invite codes before creating users.",
                created_at=20,
            )

            retrieval = RetrievalIndex(paths)
            retrieval.ensure_workspace_index(force=True)
            memory_results = retrieval.query("cedar token signing session cookies", limit=5)
            artifact_results = retrieval.query("validate invite codes before creating users", limit=5)

            self.assertTrue(any(result.source_type == "memory" and result.source_id == "fact-1" for result in memory_results))
            self.assertTrue(any(result.source_type == "artifact" and result.source_id == "artifact-1" for result in artifact_results))
