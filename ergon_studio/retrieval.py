from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
from uuid import NAMESPACE_URL
from uuid import uuid5
import time

from qdrant_client import QdrantClient
from qdrant_client import models

from ergon_studio.paths import StudioPaths


COLLECTION_NAME = "workspace_chunks"
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"
IGNORED_DIRECTORY_NAMES = {
    ".ergon.studio",
    ".git",
    ".next",
    ".nuxt",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "venv",
}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_FILE_BYTES = 128_000
TEXT_SAMPLE_BYTES = 8_192


@dataclass(frozen=True)
class RetrievalResult:
    path: str
    chunk_id: str
    text: str
    score: float
    start_line: int
    end_line: int


class RetrievalIndex:
    def __init__(
        self,
        paths: StudioPaths,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.paths = paths
        self.embedding_model = embedding_model
        self.index_path = self.paths.indexes_dir / "qdrant"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.paths.indexes_dir / "qdrant-manifest.json"
        self.embedding_cache_dir = Path.home() / ".ergon.studio" / "cache" / "fastembed"
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            client = QdrantClient(path=str(self.index_path))
            client.set_model(
                self.embedding_model,
                cache_dir=str(self.embedding_cache_dir),
                threads=1,
                providers=["CPUExecutionProvider"],
                lazy_load=True,
            )
            self._client = client
        return self._client

    def rebuild_workspace_index(self) -> int:
        workspace_fingerprint = self.workspace_fingerprint()
        documents: list[str] = []
        metadata: list[dict[str, object]] = []
        ids: list[str] = []

        for path in _workspace_files(self.paths.project_root):
            relative_path = str(path.relative_to(self.paths.project_root))
            text = _read_text_if_supported(path)
            if text is None:
                continue
            for chunk_number, chunk in enumerate(_chunk_text(text), start=1):
                chunk_id = _chunk_id(relative_path, chunk_number, chunk.text)
                documents.append(chunk.text)
                metadata.append(
                    {
                        "path": relative_path,
                        "chunk_id": chunk_id,
                        "document": chunk.text,
                        "text": chunk.text,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }
                )
                ids.append(_point_id(chunk_id))

        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        if not documents:
            return 0

        vector_name = self.client.get_vector_field_name()
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=self.client.get_fastembed_vector_params(on_disk=True),
        )
        self.client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=[
                {
                    vector_name: models.Document(
                        text=document,
                        model=self.embedding_model,
                    )
                }
                for document in documents
            ],
            payload=metadata,
            ids=ids,
            batch_size=32,
            parallel=1,
            wait=True,
        )
        self._write_manifest(
            workspace_fingerprint=workspace_fingerprint,
            chunk_count=len(documents),
            indexed_at=int(time.time()),
        )
        return len(documents)

    def ensure_workspace_index(self, *, force: bool = False) -> int:
        workspace_fingerprint = self.workspace_fingerprint()
        manifest = self._read_manifest()
        if not force and self.client.collection_exists(COLLECTION_NAME) and manifest is not None:
            if (
                manifest.get("workspace_fingerprint") == workspace_fingerprint
                and manifest.get("embedding_model") == self.embedding_model
            ):
                chunk_count = manifest.get("chunk_count", 0)
                return int(chunk_count) if isinstance(chunk_count, int) else 0
        return self.rebuild_workspace_index()

    def query(self, text: str, *, limit: int = 5) -> list[RetrievalResult]:
        query = text.strip()
        if not query:
            return []
        self.ensure_workspace_index()
        if not self.client.collection_exists(COLLECTION_NAME):
            return []
        responses = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.Document(
                text=query,
                model=self.embedding_model,
            ),
            using=self.client.get_vector_field_name(),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[RetrievalResult] = []
        for item in responses.points:
            metadata = item.payload or {}
            results.append(
                RetrievalResult(
                    path=str(metadata.get("path", "")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    text=str(metadata.get("document", metadata.get("text", ""))),
                    score=float(item.score),
                    start_line=int(metadata.get("start_line", 1)),
                    end_line=int(metadata.get("end_line", 1)),
                )
            )
        return results

    def workspace_fingerprint(self) -> str:
        digest = sha1()
        for path in _workspace_files(self.paths.project_root):
            if _read_text_if_supported(path) is None:
                continue
            stat = path.stat()
            digest.update(str(path.relative_to(self.paths.project_root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _read_manifest(self) -> dict[str, object] | None:
        if not self.manifest_path.exists():
            return None
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _write_manifest(
        self,
        *,
        workspace_fingerprint: str,
        chunk_count: int,
        indexed_at: int,
    ) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "workspace_fingerprint": workspace_fingerprint,
                    "embedding_model": self.embedding_model,
                    "chunk_count": chunk_count,
                    "indexed_at": indexed_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class _Chunk:
    text: str
    start_line: int
    end_line: int


def _workspace_files(project_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(project_root.rglob("*"))
        if path.is_file() and not any(part in IGNORED_DIRECTORY_NAMES for part in path.parts)
    ]


def _read_text_if_supported(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
        sample = raw[:TEXT_SAMPLE_BYTES]
        if b"\x00" in sample:
            return None
        return raw.decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _chunk_text(text: str) -> list[_Chunk]:
    if not text.strip():
        return []
    lines = text.splitlines()
    chunks: list[_Chunk] = []
    current_lines: list[str] = []
    current_length = 0
    start_line = 1

    for index, line in enumerate(lines, start=1):
        line_length = len(line) + 1
        if current_lines and current_length + line_length > CHUNK_SIZE:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(_Chunk(text=chunk_text, start_line=start_line, end_line=index - 1))
            overlap_lines = _overlap_tail(current_lines)
            current_lines = overlap_lines + [line]
            current_length = sum(len(item) + 1 for item in current_lines)
            start_line = max(1, index - len(overlap_lines))
            continue
        if not current_lines:
            start_line = index
        current_lines.append(line)
        current_length += line_length

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(_Chunk(text=chunk_text, start_line=start_line, end_line=len(lines)))
    return chunks


def _overlap_tail(lines: list[str]) -> list[str]:
    if not lines:
        return []
    selected: list[str] = []
    total = 0
    for line in reversed(lines):
        selected.append(line)
        total += len(line) + 1
        if total >= CHUNK_OVERLAP:
            break
    return list(reversed(selected))


def _chunk_id(path: str, chunk_number: int, text: str) -> str:
    digest = sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{path}:{chunk_number}:{digest}"


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))
