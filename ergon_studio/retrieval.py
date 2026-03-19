from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL
from uuid import uuid5
import time

import portalocker
from qdrant_client import QdrantClient
from qdrant_client import models

from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.memory_store import MemoryStore
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
    source_type: str = "workspace"
    source_id: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class _IndexSource:
    key: str
    source_type: str
    source_id: str
    path: str
    text: str
    fingerprint: str
    title: str | None = None


class RetrievalIndex:
    def __init__(
        self,
        paths: StudioPaths,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.paths = paths
        self.embedding_model = embedding_model
        self.memory_store = MemoryStore(paths)
        self.artifact_store = ArtifactStore(paths)
        self.index_path = self.paths.indexes_dir / "qdrant"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.paths.indexes_dir / "qdrant-client.lock"
        self.manifest_path = self.paths.indexes_dir / "qdrant-manifest.json"
        self.embedding_cache_dir = Path.home() / ".ergon.studio" / "cache" / "fastembed"
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _client_session(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(self.lock_path, timeout=60):
            client = QdrantClient(path=str(self.index_path))
            client.set_model(
                self.embedding_model,
                cache_dir=str(self.embedding_cache_dir),
                threads=1,
                providers=["CPUExecutionProvider"],
                lazy_load=True,
            )
            try:
                yield client
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    def rebuild_workspace_index(self) -> int:
        with self._client_session() as client:
            return self._rebuild_workspace_index(client)

    def ensure_workspace_index(self, *, force: bool = False) -> int:
        with self._client_session() as client:
            return self._ensure_workspace_index(client, force=force)

    def query(self, text: str, *, limit: int = 5) -> list[RetrievalResult]:
        return self.query_many([text], limit=limit)

    def query_many(self, texts: list[str], *, limit: int = 5) -> list[RetrievalResult]:
        queries = _normalize_queries(texts)
        if not queries:
            return []
        with self._client_session() as client:
            self._ensure_workspace_index(client)
            if not client.collection_exists(COLLECTION_NAME):
                return []
            merged: dict[str, RetrievalResult] = {}
            per_query_limit = max(limit, 4)
            for query in queries:
                responses = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=models.Document(
                        text=query,
                        model=self.embedding_model,
                    ),
                    using=client.get_vector_field_name(),
                    limit=per_query_limit,
                    with_payload=True,
                    with_vectors=False,
                )
                for item in responses.points:
                    result = _result_from_payload(item.payload or {}, score=float(item.score))
                    current = merged.get(result.chunk_id)
                    if current is None or result.score > current.score:
                        merged[result.chunk_id] = result
            return sorted(
                merged.values(),
                key=lambda item: (-item.score, item.path, item.chunk_id),
            )[:limit]

    def _rebuild_workspace_index(self, client: QdrantClient) -> int:
        sources = self._collect_sources()
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        if not sources:
            self._write_manifest(
                workspace_fingerprint=self._sources_fingerprint({}),
                chunk_count=0,
                indexed_at=int(time.time()),
                source_entries={},
            )
            return 0

        self._ensure_collection(client)
        source_entries: dict[str, dict[str, Any]] = {}
        for source in sources.values():
            point_ids = self._index_source(client, source)
            source_entries[source.key] = self._manifest_source_entry(source, point_ids)
        chunk_count = sum(len(entry["point_ids"]) for entry in source_entries.values())
        self._write_manifest(
            workspace_fingerprint=self._sources_fingerprint(sources),
            chunk_count=chunk_count,
            indexed_at=int(time.time()),
            source_entries=source_entries,
        )
        return chunk_count

    def _ensure_workspace_index(self, client: QdrantClient, *, force: bool = False) -> int:
        sources = self._collect_sources()
        workspace_fingerprint = self._sources_fingerprint(sources)
        manifest = self._read_manifest()
        if force:
            return self._rebuild_workspace_index(client)
        if not sources:
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
            self._write_manifest(
                workspace_fingerprint=workspace_fingerprint,
                chunk_count=0,
                indexed_at=int(time.time()),
                source_entries={},
            )
            return 0
        if not client.collection_exists(COLLECTION_NAME) or manifest is None:
            return self._rebuild_workspace_index(client)
        if manifest.get("embedding_model") != self.embedding_model:
            return self._rebuild_workspace_index(client)
        if manifest.get("workspace_fingerprint") == workspace_fingerprint:
            chunk_count = manifest.get("chunk_count", 0)
            return int(chunk_count) if isinstance(chunk_count, int) else 0
        return self._sync_workspace_index(client=client, sources=sources, manifest=manifest)

    def workspace_fingerprint(self) -> str:
        return self._sources_fingerprint(self._collect_sources())

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
        source_entries: dict[str, dict[str, Any]],
    ) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "workspace_fingerprint": workspace_fingerprint,
                    "embedding_model": self.embedding_model,
                    "chunk_count": chunk_count,
                    "sources": source_entries,
                    "indexed_at": indexed_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _collect_sources(self) -> dict[str, _IndexSource]:
        sources: dict[str, _IndexSource] = {}
        for path in _workspace_files(self.paths.project_root):
            text = _read_text_if_supported(path)
            if text is None or not text.strip():
                continue
            relative_path = str(path.relative_to(self.paths.project_root))
            source_key = f"workspace:{relative_path}"
            stat = path.stat()
            sources[source_key] = _IndexSource(
                key=source_key,
                source_type="workspace",
                source_id=relative_path,
                path=relative_path,
                text=text,
                fingerprint=_fingerprint_parts(relative_path, str(stat.st_size), str(stat.st_mtime_ns)),
            )

        for fact in self.memory_store.list_facts(scopes=("project", "user")):
            text = _render_memory_fact(fact)
            if not text.strip():
                continue
            source_key = f"memory:{fact.id}"
            sources[source_key] = _IndexSource(
                key=source_key,
                source_type="memory",
                source_id=fact.id,
                path=f"memory/{fact.scope}/{fact.kind}/{fact.id}",
                text=text,
                fingerprint=_fingerprint_parts(
                    fact.id,
                    fact.scope,
                    fact.kind,
                    fact.content,
                    fact.source or "",
                    ",".join(fact.tags),
                    str(fact.last_used_at or 0),
                ),
                title=f"{fact.kind} [{fact.scope}]",
            )

        for artifact in self.artifact_store.list_all_artifacts():
            body = self.artifact_store.read_artifact_body(artifact)
            if not body.strip():
                continue
            source_key = f"artifact:{artifact.id}"
            sources[source_key] = _IndexSource(
                key=source_key,
                source_type="artifact",
                source_id=artifact.id,
                path=f"artifacts/{artifact.kind}/{artifact.id}",
                text=_render_artifact(artifact.title, artifact.kind, body),
                fingerprint=_fingerprint_parts(
                    artifact.id,
                    artifact.kind,
                    artifact.title,
                    body,
                    str(artifact.created_at),
                ),
                title=artifact.title,
            )
        return sources

    def _sync_workspace_index(
        self,
        *,
        client: QdrantClient,
        sources: dict[str, _IndexSource],
        manifest: dict[str, object],
    ) -> int:
        self._ensure_collection(client)
        old_entries = manifest.get("sources", {})
        if not isinstance(old_entries, dict):
            return self._rebuild_workspace_index(client)

        removed_keys = sorted(set(old_entries) - set(sources))
        changed_keys = sorted(
            key
            for key, source in sources.items()
            if not isinstance(old_entries.get(key), dict)
            or old_entries[key].get("fingerprint") != source.fingerprint
        )

        for key in removed_keys + changed_keys:
            self._delete_point_ids(client, _manifest_point_ids(old_entries.get(key)))

        source_entries: dict[str, dict[str, Any]] = {}
        for key, source in sources.items():
            if key in changed_keys:
                point_ids = self._index_source(client, source)
                source_entries[key] = self._manifest_source_entry(source, point_ids)
                continue
            entry = old_entries.get(key)
            if not isinstance(entry, dict):
                point_ids = self._index_source(client, source)
                source_entries[key] = self._manifest_source_entry(source, point_ids)
                continue
            source_entries[key] = entry

        chunk_count = sum(len(_manifest_point_ids(entry)) for entry in source_entries.values())
        self._write_manifest(
            workspace_fingerprint=self._sources_fingerprint(sources),
            chunk_count=chunk_count,
            indexed_at=int(time.time()),
            source_entries=source_entries,
        )
        return chunk_count

    def _ensure_collection(self, client: QdrantClient) -> None:
        if client.collection_exists(COLLECTION_NAME):
            return
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=client.get_fastembed_vector_params(on_disk=True),
        )

    def _index_source(self, client: QdrantClient, source: _IndexSource) -> list[str]:
        chunks = _chunk_text(source.text)
        if not chunks:
            return []
        ids: list[str] = []
        metadata: list[dict[str, object]] = []
        vectors: list[dict[str, models.Document]] = []
        vector_name = client.get_vector_field_name()
        for chunk_number, chunk in enumerate(chunks, start=1):
            chunk_id = _chunk_id(source.key, chunk_number, chunk.text)
            point_id = _point_id(chunk_id)
            ids.append(point_id)
            vectors.append(
                {
                    vector_name: models.Document(
                        text=chunk.text,
                        model=self.embedding_model,
                    )
                }
            )
            payload: dict[str, object] = {
                "path": source.path,
                "chunk_id": chunk_id,
                "document": chunk.text,
                "text": chunk.text,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "source_type": source.source_type,
                "source_id": source.source_id,
            }
            if source.title is not None:
                payload["title"] = source.title
            metadata.append(payload)
        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=vectors,
            payload=metadata,
            ids=ids,
            batch_size=32,
            parallel=1,
            wait=True,
        )
        return ids

    def _delete_point_ids(self, client: QdrantClient, point_ids: list[str]) -> None:
        if not point_ids:
            return
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

    def _manifest_source_entry(self, source: _IndexSource, point_ids: list[str]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "fingerprint": source.fingerprint,
            "path": source.path,
            "point_ids": point_ids,
            "source_id": source.source_id,
            "source_type": source.source_type,
        }
        if source.title is not None:
            entry["title"] = source.title
        return entry

    def _sources_fingerprint(self, sources: dict[str, _IndexSource]) -> str:
        digest = sha1()
        for key in sorted(sources):
            source = sources[key]
            digest.update(source.key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.fingerprint.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


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


def _fingerprint_parts(*parts: str) -> str:
    digest = sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_point_ids(entry: object) -> list[str]:
    if not isinstance(entry, dict):
        return []
    point_ids = entry.get("point_ids", [])
    if not isinstance(point_ids, list):
        return []
    return [str(point_id) for point_id in point_ids]


def _normalize_queries(texts: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for text in texts:
        query = text.strip()
        if not query or query in seen:
            continue
        seen.add(query)
        normalized.append(query)
    return normalized


def _render_memory_fact(fact) -> str:
    lines = [fact.content.strip()]
    lines.append(f"Scope: {fact.scope}")
    lines.append(f"Kind: {fact.kind}")
    if fact.source:
        lines.append(f"Source: {fact.source}")
    if fact.tags:
        lines.append(f"Tags: {', '.join(fact.tags)}")
    return "\n".join(line for line in lines if line.strip()).strip()


def _render_artifact(title: str, kind: str, body: str) -> str:
    lines = [title.strip(), f"Kind: {kind}", body.strip()]
    return "\n\n".join(line for line in lines if line.strip()).strip()


def _result_from_payload(payload: dict[str, object], *, score: float) -> RetrievalResult:
    return RetrievalResult(
        path=str(payload.get("path", "")),
        chunk_id=str(payload.get("chunk_id", "")),
        text=str(payload.get("document", payload.get("text", ""))),
        score=score,
        start_line=int(payload.get("start_line", 1)),
        end_line=int(payload.get("end_line", 1)),
        source_type=str(payload.get("source_type", "workspace")),
        source_id=str(payload["source_id"]) if payload.get("source_id") is not None else None,
        title=str(payload["title"]) if payload.get("title") is not None else None,
    )
