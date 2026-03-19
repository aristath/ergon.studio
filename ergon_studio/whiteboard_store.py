from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.sqlite import MetadataStore
from ergon_studio.storage.models import validate_unix_time


WHITEBOARD_SECTION_TITLES = (
    "Goal",
    "Constraints",
    "Plan",
    "Decisions",
    "Open Questions",
    "Acceptance Criteria",
)


@dataclass(frozen=True)
class TaskWhiteboardRecord:
    task_id: str
    title: str
    updated_at: int
    file_path: Path
    body: str
    sections: dict[str, str]
    parent_task_id: str | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.updated_at, "updated_at")
        if not isinstance(self.file_path, Path):
            raise TypeError("file_path must be a Path")


class WhiteboardStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def whiteboard_path(self, task_id: str, *, session_id: str | None = None) -> Path:
        resolved_session_id = session_id
        if resolved_session_id is None:
            task = self.metadata.get_task(task_id)
            resolved_session_id = task.session_id if task is not None else None
        if resolved_session_id is None:
            return self.paths.whiteboards_dir / f"{task_id}.md"
        return self.paths.session_whiteboards_dir(resolved_session_id) / f"{task_id}.md"

    def read_task_whiteboard(self, task_id: str) -> TaskWhiteboardRecord | None:
        path = self.whiteboard_path(task_id)
        if not path.exists():
            return None
        return parse_task_whiteboard_text(path.read_text(encoding="utf-8"), path=path)

    def read_task_whiteboard_text(self, task_id: str) -> str:
        path = self.whiteboard_path(task_id)
        if not path.exists():
            raise ValueError(f"unknown task whiteboard: {task_id}")
        return path.read_text(encoding="utf-8")

    def save_task_whiteboard_text(self, task_id: str, text: str) -> TaskWhiteboardRecord:
        path = self.whiteboard_path(task_id)
        record = parse_task_whiteboard_text(text, path=path)
        if record.task_id != task_id:
            raise ValueError("whiteboard task_id must match the selected task")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_ensure_trailing_newline(text), encoding="utf-8")
        return record

    def ensure_task_whiteboard(
        self,
        *,
        task_id: str,
        title: str,
        updated_at: int,
        parent_task_id: str | None = None,
        goal: str | None = None,
        template_task_id: str | None = None,
    ) -> TaskWhiteboardRecord:
        existing = self.read_task_whiteboard(task_id)
        if existing is not None:
            return existing

        sections = {title: "" for title in WHITEBOARD_SECTION_TITLES}
        if template_task_id is not None:
            template = self.read_task_whiteboard(template_task_id)
            if template is not None:
                for title_key in WHITEBOARD_SECTION_TITLES:
                    sections[title_key] = template.sections.get(title_key, "")
        if goal:
            sections["Goal"] = goal.strip()
        elif not sections["Goal"]:
            sections["Goal"] = title

        return self.write_task_whiteboard(
            task_id=task_id,
            title=title,
            updated_at=updated_at,
            parent_task_id=parent_task_id,
            sections=sections,
        )

    def update_task_whiteboard(
        self,
        *,
        task_id: str,
        updated_at: int,
        title: str | None = None,
        parent_task_id: str | None = None,
        section_updates: dict[str, str] | None = None,
    ) -> TaskWhiteboardRecord:
        existing = self.read_task_whiteboard(task_id)
        if existing is None:
            raise ValueError(f"unknown task whiteboard: {task_id}")

        sections = dict(existing.sections)
        if section_updates:
            for section_title, value in section_updates.items():
                if section_title not in WHITEBOARD_SECTION_TITLES:
                    raise ValueError(f"unknown whiteboard section: {section_title}")
                sections[section_title] = value.strip()

        return self.write_task_whiteboard(
            task_id=task_id,
            title=title or existing.title,
            updated_at=updated_at,
            parent_task_id=parent_task_id if parent_task_id is not None else existing.parent_task_id,
            sections=sections,
        )

    def write_task_whiteboard(
        self,
        *,
        task_id: str,
        title: str,
        updated_at: int,
        sections: dict[str, str],
        parent_task_id: str | None = None,
    ) -> TaskWhiteboardRecord:
        path = self.whiteboard_path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "task_id": task_id,
            "title": title,
            "updated_at": updated_at,
        }
        if parent_task_id is not None:
            metadata["parent_task_id"] = parent_task_id
        body = render_whiteboard_body(sections)
        text = render_task_whiteboard_text(metadata=metadata, body=body)
        path.write_text(text, encoding="utf-8")
        return parse_task_whiteboard_text(text, path=path)


def render_task_whiteboard_text(*, metadata: dict[str, Any], body: str) -> str:
    frontmatter = yaml.safe_dump(metadata, sort_keys=False).strip()
    return f"---\n{frontmatter}\n---\n{body.strip()}\n"


def render_whiteboard_body(sections: dict[str, str]) -> str:
    parts: list[str] = []
    for section_title in WHITEBOARD_SECTION_TITLES:
        parts.append(f"## {section_title}")
        value = sections.get(section_title, "").strip()
        if value:
            parts.append(value)
        parts.append("")
    return "\n".join(parts).strip()


def parse_task_whiteboard_text(text: str, *, path: Path) -> TaskWhiteboardRecord:
    metadata_text, body = _split_frontmatter(text)
    metadata = yaml.safe_load(metadata_text) if metadata_text else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} frontmatter must be a mapping")
    if "task_id" not in metadata:
        raise ValueError(f"{path} frontmatter must include task_id")
    if "title" not in metadata:
        raise ValueError(f"{path} frontmatter must include title")
    if "updated_at" not in metadata:
        raise ValueError(f"{path} frontmatter must include updated_at")

    updated_at = metadata["updated_at"]
    validate_unix_time(updated_at, "updated_at")
    stripped_body = body.strip()
    sections = _parse_sections(stripped_body)
    for section_title in WHITEBOARD_SECTION_TITLES:
        sections.setdefault(section_title, "")

    return TaskWhiteboardRecord(
        task_id=str(metadata["task_id"]),
        title=str(metadata["title"]),
        updated_at=updated_at,
        file_path=path,
        body=stripped_body,
        sections=sections,
        parent_task_id=str(metadata["parent_task_id"]) if metadata.get("parent_task_id") is not None else None,
    )


def _split_frontmatter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return frontmatter, body

    raise ValueError("frontmatter opening marker found without a closing marker")


def _parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line[3:].strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"
