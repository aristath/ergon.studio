from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DefinitionDocument:
    id: str
    path: Path
    metadata: dict[str, Any]
    body: str
    sections: dict[str, str]


def load_definition(path: Path) -> DefinitionDocument:
    text = path.read_text(encoding="utf-8")
    return parse_definition_text(text, path=path)


def load_definitions_from_dir(directory: Path) -> dict[str, DefinitionDocument]:
    definitions: dict[str, DefinitionDocument] = {}
    if not directory.exists():
        return definitions

    for path in sorted(directory.glob("*.md")):
        definition = load_definition(path)
        if definition.id in definitions:
            raise ValueError(f"duplicate definition id: {definition.id}")
        definitions[definition.id] = definition
    return definitions


def parse_definition_text(text: str, *, path: Path) -> DefinitionDocument:
    frontmatter_text, body = _split_frontmatter(text)
    metadata = yaml.safe_load(frontmatter_text) if frontmatter_text else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} frontmatter must be a mapping")
    if "id" not in metadata:
        raise ValueError(f"{path} frontmatter must include an id")

    body = body.strip()
    return DefinitionDocument(
        id=str(metadata["id"]),
        path=path,
        metadata=metadata,
        body=body,
        sections=_parse_sections(body),
    )


def format_definition_section(
    definition: DefinitionDocument,
    section: str,
    **kwargs: str,
) -> str:
    """Return the named section from a definition with {placeholders} substituted.

    Returns an empty string if the section does not exist.
    """
    template = definition.sections.get(section, "")
    return template.format_map(kwargs) if template else ""


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
