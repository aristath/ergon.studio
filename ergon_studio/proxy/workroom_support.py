from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.prompts import (
    summary_instructions,
    workroom_summary_prompt,
)
from ergon_studio.proxy.turn_state import ProxyTurnState

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyWorkroomSupport:
    def __init__(
        self,
        *,
        run_text_agent: Callable[..., Any],
    ) -> None:
        self._run_text_agent = run_text_agent

    async def emit_summary(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        current_brief: str,
        workroom_outputs: tuple[str, ...],
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=workroom_summary_prompt(
                workroom_id=definition.id,
                goal=goal,
                outputs=workroom_outputs,
            ),
            preamble=summary_instructions(),
            session_id=f"proxy-workroom-summary-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not final_text:
            final_text = current_brief.strip()
        state.set_content(final_text)
        if final_text:
            yield ProxyContentDeltaEvent(final_text)
