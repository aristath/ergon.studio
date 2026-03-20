from __future__ import annotations

from agent_framework import BaseContextProvider

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.registry import RuntimeRegistry


class AgentProfileContextProvider(BaseContextProvider):
    def __init__(
        self,
        definition: DefinitionDocument,
        *,
        registry: RuntimeRegistry | None = None,
        provider_name: str | None = None,
        provider_capabilities: dict[str, object] | None = None,
    ) -> None:
        super().__init__("agent_profile")
        self.definition = definition
        self.registry = registry
        self.provider_name = provider_name
        self.provider_capabilities = provider_capabilities or {}

    async def before_run(self, *, agent, session, context, state) -> None:
        role = str(self.definition.metadata.get("role", self.definition.id))
        tools = self.definition.metadata.get("tools", [])
        flags = []
        for key in ("can_speak_unprompted", "can_interrupt_on_risk", "can_propose_replan", "can_request_user_input"):
            if key in self.definition.metadata:
                flags.append(f"{key}={self.definition.metadata[key]}")
        lines = [
            f"Agent profile: {self.definition.id}",
            f"Role: {role}",
            f"Tools: {', '.join(str(tool) for tool in tools) if isinstance(tools, list) and tools else 'none'}",
        ]
        if self.provider_name is not None:
            capability_text = ", ".join(
                f"{key}={value}" for key, value in sorted(self.provider_capabilities.items())
            ) or "none"
            lines.append(f"Provider: {self.provider_name}")
            lines.append(f"Provider capabilities: {capability_text}")
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
            for workflow_id, definition in sorted(self.registry.workflow_definitions.items()):
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
