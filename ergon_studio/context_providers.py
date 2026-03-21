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
from ergon_studio.workroom_layout import workroom_kind_for_definition


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
        flags = []
        for key in (
            "can_speak_unprompted",
            "can_interrupt_on_risk",
            "can_propose_replan",
            "can_request_user_input",
        ):
            if key in self.definition.metadata:
                flags.append(f"{key}={self.definition.metadata[key]}")
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
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
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
                kind = workroom_kind_for_definition(definition)
                workroom_summaries.append(f"{workroom_id}({kind})")
            lines.append(
                "Available specialists: "
                + (", ".join(agent_summaries) if agent_summaries else "none")
            )
            lines.append(
                "Available workrooms: "
                + (", ".join(workroom_summaries) if workroom_summaries else "none")
            )
        context.extend_instructions(self.source_id, "\n".join(lines))
