from __future__ import annotations

from typing import Any

from agent_framework import (
    AgentSession,
    BaseContextProvider,
    SessionContext,
    SupportsAgentRun,
)

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.workroom_layout import workroom_participants_for_definition


class AgentProfileContextProvider(BaseContextProvider):
    def __init__(
        self,
        definition: DefinitionDocument,
        *,
        registry: RuntimeRegistry | None = None,
    ) -> None:
        super().__init__("agent_profile")
        self.definition = definition
        self.registry = registry

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        del agent, session, state
        role = str(self.definition.metadata.get("role", self.definition.id))
        tools = self.definition.metadata.get("tools", [])
        tool_summary = (
            ", ".join(str(tool) for tool in tools)
            if isinstance(tools, list) and tools
            else "none"
        )
        lines = [
            f"Agent profile: {self.definition.id}",
            f"Role: {role}",
            f"Tools: {tool_summary}",
        ]
        if role == "orchestrator" and self.registry is not None:
            agent_summaries = []
            for agent_id, definition in sorted(self.registry.agent_definitions.items()):
                if agent_id == self.definition.id:
                    continue
                agent_role = str(definition.metadata.get("role", agent_id))
                agent_summaries.append(f"{agent_id}({agent_role})")
            workroom_summaries = []
            for workroom_id, definition in sorted(
                self.registry.workroom_definitions.items()
            ):
                participants = ", ".join(
                    workroom_participants_for_definition(definition)
                )
                workroom_summaries.append(f"{workroom_id}({participants})")
            lines.append(
                "Available specialists: "
                + (", ".join(agent_summaries) if agent_summaries else "none")
            )
            lines.append(
                "Available workroom presets: "
                + (", ".join(workroom_summaries) if workroom_summaries else "none")
            )
        context.extend_instructions(self.source_id, "\n".join(lines))
