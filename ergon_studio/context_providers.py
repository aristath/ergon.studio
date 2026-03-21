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
            workflow_summaries = []
            for workflow_id, definition in sorted(
                self.registry.workflow_definitions.items()
            ):
                orchestration = str(definition.metadata.get("orchestration", "unknown"))
                workflow_summaries.append(f"{workflow_id}({orchestration})")
            lines.append(
                "Available specialists: "
                + (", ".join(agent_summaries) if agent_summaries else "none")
            )
            lines.append(
                "Available workflows: "
                + (", ".join(workflow_summaries) if workflow_summaries else "none")
            )
        context.extend_instructions(self.source_id, "\n".join(lines))
