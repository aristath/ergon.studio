from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
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
