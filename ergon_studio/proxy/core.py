from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any
from uuid import uuid4

from agent_framework import Message, ResponseStream

from ergon_studio.agent_factory import build_agent
from ergon_studio.proxy.continuation import ContinuationState, encode_continuation_tool_call, latest_continuation
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyReasoningDeltaEvent, ProxyToolCallEvent, ProxyTurnResult
from ergon_studio.proxy.planner import ProxyTurnPlan, build_turn_planner_instructions, build_turn_planner_prompt, parse_turn_plan, summarize_conversation
from ergon_studio.proxy.tool_policy import resolve_agent_tool_policy
from ergon_studio.proxy.tool_passthrough import build_declaration_tools, extract_tool_calls
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.workflow_compiler import workflow_step_groups_for_definition


class ProxyOrchestrationCore:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_builder: Callable[..., Any] = build_agent,
    ) -> None:
        self.registry = registry
        self._agent_builder = agent_builder

    def stream_turn(
        self,
        request,
        *,
        created_at: int | None = None,
    ) -> ResponseStream[ProxyReasoningDeltaEvent | ProxyContentDeltaEvent | ProxyToolCallEvent | ProxyFinishEvent, ProxyTurnResult]:
        if created_at is None:
            created_at = int(time.time())
        state: dict[str, Any] = {
            "content": "",
            "reasoning": "",
            "mode": "act",
            "finish_reason": "stop",
            "tool_calls": (),
        }

        async def _events():
            continuation = latest_continuation(request.messages)
            if continuation is not None:
                state["mode"] = continuation.mode
                async for event in self._execute_continuation(
                    request=request,
                    continuation=continuation,
                    created_at=created_at,
                    state=state,
                ):
                    yield event
            else:
                plan = await self._plan_turn(request)
                state["mode"] = plan.mode
                async for event in self._execute_plan(
                    request=request,
                    plan=plan,
                    created_at=created_at,
                    state=state,
                ):
                    yield event
            yield ProxyFinishEvent(state["finish_reason"])

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason=state["finish_reason"],
                content=state["content"],
                reasoning=state["reasoning"],
                mode=state["mode"],
                tool_calls=state["tool_calls"],
            ),
        )

    async def _plan_turn(self, request) -> ProxyTurnPlan:
        planner_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=build_turn_planner_prompt(request),
            preamble=build_turn_planner_instructions(self.registry),
            session_id=f"proxy-planner-{uuid4().hex}",
        )
        if not planner_text:
            return ProxyTurnPlan(mode="act")
        try:
            return parse_turn_plan(planner_text, registry=self.registry)
        except ValueError:
            return ProxyTurnPlan(mode="act")

    async def _execute_plan(self, *, request, plan: ProxyTurnPlan, created_at: int, state: dict[str, Any]):
        if plan.mode == "delegate" and plan.agent_id is not None:
            async for event in self._execute_delegation(request=request, plan=plan, created_at=created_at, state=state):
                yield event
            return
        if plan.mode == "workflow" and plan.workflow_id is not None:
            async for event in self._execute_workflow(request=request, plan=plan, created_at=created_at, state=state):
                yield event
            return
        async for event in self._execute_direct(request=request, created_at=created_at, state=state):
            yield event

    async def _execute_continuation(
        self,
        *,
        request,
        continuation: ContinuationState,
        created_at: int,
        state: dict[str, Any],
    ):
        if continuation.mode == "workflow" and continuation.workflow_id is not None:
            async for event in self._execute_workflow_continuation(
                request=request,
                continuation=continuation,
                created_at=created_at,
                state=state,
            ):
                yield event
            return
        if continuation.mode == "delegate":
            plan = ProxyTurnPlan(
                mode="delegate",
                agent_id=continuation.agent_id,
                request=continuation.request_text or request.latest_user_text(),
            )
            async for event in self._execute_delegation(
                request=request,
                plan=plan,
                created_at=created_at,
                state=state,
                current_brief=continuation.current_brief,
            ):
                yield event
            return
        async for event in self._execute_direct(request=request, created_at=created_at, state=state):
            yield event

    async def _execute_direct(self, *, request, created_at: int, state: dict[str, Any]):
        notice = "Orchestrator: handling this turn directly.\n"
        state["reasoning"] += notice
        yield ProxyReasoningDeltaEvent(notice)
        prompt = _direct_reply_prompt(request)
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id="orchestrator",
            prompt=prompt,
            session_id=f"proxy-direct-{uuid4().hex}",
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            final_response_sink=lambda value: _set_response_holder(response_holder, value),
        ):
            state["content"] += delta
            yield ProxyContentDeltaEvent(delta)
        response = response_holder.get("response")
        if response is not None:
            emitted = self._emit_tool_calls(
                response=response,
                continuation=ContinuationState(mode="act", agent_id="orchestrator"),
                state=state,
            )
            for event in emitted:
                yield event

    async def _execute_delegation(
        self,
        *,
        request,
        plan: ProxyTurnPlan,
        created_at: int,
        state: dict[str, Any],
        current_brief: str | None = None,
    ):
        agent_id = plan.agent_id or "coder"
        intro = f"Orchestrator: delegating this turn to {agent_id}.\n"
        state["reasoning"] += intro
        yield ProxyReasoningDeltaEvent(intro)
        specialist_prompt = _specialist_prompt(
            specialist_id=agent_id,
            request_text=plan.request or request.latest_user_text() or "",
            transcript_summary=summarize_conversation(request.messages),
            current_brief=current_brief,
        )
        specialist_text = ""
        first = True
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id=agent_id,
            prompt=specialist_prompt,
            session_id=f"proxy-delegate-{agent_id}-{uuid4().hex}",
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            final_response_sink=lambda value: _set_response_holder(response_holder, value),
        ):
            specialist_text += delta
            reasoning_delta = f"{agent_id}: {delta}" if first else delta
            first = False
            state["reasoning"] += reasoning_delta
            yield ProxyReasoningDeltaEvent(reasoning_delta)
        response = response_holder.get("response")
        if response is not None:
            emitted = self._emit_tool_calls(
                response=response,
                continuation=ContinuationState(
                    mode="delegate",
                    agent_id=agent_id,
                    request_text=plan.request or request.latest_user_text(),
                    current_brief=specialist_text.strip() or current_brief,
                ),
                state=state,
            )
            if emitted:
                for event in emitted:
                    yield event
                return
        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=_delegation_summary_prompt(
                request_text=request.latest_user_text() or "",
                specialist_id=agent_id,
                specialist_text=specialist_text,
            ),
            preamble=_summary_instructions(),
            session_id=f"proxy-delegation-summary-{uuid4().hex}",
        )
        if not final_text:
            final_text = specialist_text.strip()
        state["content"] = final_text
        if final_text:
            yield ProxyContentDeltaEvent(final_text)

    async def _execute_workflow(self, *, request, plan: ProxyTurnPlan, created_at: int, state: dict[str, Any]):
        definition = self.registry.workflow_definitions.get(plan.workflow_id or "")
        if definition is None:
            state["finish_reason"] = "error"
            error_text = f"Unknown workflow: {plan.workflow_id or '(none)'}"
            state["content"] = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = f"Orchestrator: running workflow {definition.id}.\n"
        state["reasoning"] += intro
        yield ProxyReasoningDeltaEvent(intro)

        goal = plan.goal or request.latest_user_text() or ""
        step_groups = workflow_step_groups_for_definition(definition)
        current_brief = goal
        workflow_outputs: list[str] = []
        for step_index, group in enumerate(step_groups):
            for agent_id in group:
                specialist_prompt = _workflow_step_prompt(
                    workflow_id=definition.id,
                    agent_id=agent_id,
                    goal=goal,
                    current_brief=current_brief,
                    transcript_summary=summarize_conversation(request.messages),
                    prior_outputs=tuple(workflow_outputs),
                )
                agent_text = ""
                first = True
                response_holder: dict[str, Any] = {}
                async for delta in self._stream_text_agent(
                    agent_id=agent_id,
                    prompt=specialist_prompt,
                    session_id=f"proxy-workflow-{definition.id}-{agent_id}-{uuid4().hex}",
                    host_tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    final_response_sink=lambda value: _set_response_holder(response_holder, value),
                ):
                    agent_text += delta
                    reasoning_delta = f"{agent_id}: {delta}" if first else delta
                    first = False
                    state["reasoning"] += reasoning_delta
                    yield ProxyReasoningDeltaEvent(reasoning_delta)
                response = response_holder.get("response")
                if response is not None:
                    emitted = self._emit_tool_calls(
                        response=response,
                        continuation=ContinuationState(
                            mode="workflow",
                            workflow_id=definition.id,
                            step_index=step_index,
                            agent_id=agent_id,
                            goal=goal,
                            current_brief=agent_text.strip() or current_brief,
                            workflow_outputs=tuple(workflow_outputs),
                        ),
                        state=state,
                    )
                    if emitted:
                        for event in emitted:
                            yield event
                        return
                workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
                current_brief = agent_text.strip() or current_brief

        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=_workflow_summary_prompt(
                workflow_id=definition.id,
                goal=goal,
                outputs=tuple(workflow_outputs),
            ),
            preamble=_summary_instructions(),
            session_id=f"proxy-workflow-summary-{uuid4().hex}",
        )
        if not final_text:
            final_text = current_brief.strip()
        state["content"] = final_text
        if final_text:
            yield ProxyContentDeltaEvent(final_text)

    async def _execute_workflow_continuation(
        self,
        *,
        request,
        continuation: ContinuationState,
        created_at: int,
        state: dict[str, Any],
    ):
        definition = self.registry.workflow_definitions.get(continuation.workflow_id or "")
        if definition is None:
            state["finish_reason"] = "error"
            error_text = f"Unknown workflow: {continuation.workflow_id or '(none)'}"
            state["content"] = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = f"Orchestrator: continuing workflow {definition.id} with {continuation.agent_id}.\n"
        state["reasoning"] += intro
        yield ProxyReasoningDeltaEvent(intro)

        goal = continuation.goal or request.latest_user_text() or ""
        step_groups = workflow_step_groups_for_definition(definition)
        start_index = continuation.step_index or 0
        current_brief = continuation.current_brief or goal
        workflow_outputs: list[str] = list(continuation.workflow_outputs)
        for step_index in range(start_index, len(step_groups)):
            group = step_groups[step_index]
            agents = group
            if step_index == start_index:
                agents = tuple(agent for agent in group if agent == continuation.agent_id) or (continuation.agent_id,)
            for agent_id in agents:
                specialist_prompt = _workflow_step_prompt(
                    workflow_id=definition.id,
                    agent_id=agent_id,
                    goal=goal,
                    current_brief=current_brief,
                    transcript_summary=summarize_conversation(request.messages),
                    prior_outputs=tuple(workflow_outputs),
                )
                agent_text = ""
                first = True
                response_holder: dict[str, Any] = {}
                async for delta in self._stream_text_agent(
                    agent_id=agent_id,
                    prompt=specialist_prompt,
                    session_id=f"proxy-workflow-{definition.id}-{agent_id}-{uuid4().hex}",
                    host_tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    final_response_sink=lambda value: _set_response_holder(response_holder, value),
                ):
                    agent_text += delta
                    reasoning_delta = f"{agent_id}: {delta}" if first else delta
                    first = False
                    state["reasoning"] += reasoning_delta
                    yield ProxyReasoningDeltaEvent(reasoning_delta)
                response = response_holder.get("response")
                if response is not None:
                    emitted = self._emit_tool_calls(
                        response=response,
                        continuation=ContinuationState(
                            mode="workflow",
                            workflow_id=definition.id,
                            step_index=step_index,
                            agent_id=agent_id,
                            goal=goal,
                            current_brief=agent_text.strip() or current_brief,
                            workflow_outputs=tuple(workflow_outputs),
                        ),
                        state=state,
                    )
                    if emitted:
                        for event in emitted:
                            yield event
                        return
                workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
                current_brief = agent_text.strip() or current_brief

        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=_workflow_summary_prompt(
                workflow_id=definition.id,
                goal=goal,
                outputs=tuple(workflow_outputs),
            ),
            preamble=_summary_instructions(),
            session_id=f"proxy-workflow-summary-{uuid4().hex}",
        )
        if not final_text:
            final_text = current_brief.strip()
        state["content"] = final_text
        if final_text:
            yield ProxyContentDeltaEvent(final_text)

    async def _run_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        preamble: str = "",
    ) -> str | None:
        full_prompt = _merge_preamble(preamble, prompt)
        response = await self._run_agent(agent_id=agent_id, prompt=full_prompt, session_id=session_id, stream=False)
        if response is None:
            return None
        response_text = getattr(response, "text", response)
        if not isinstance(response_text, str):
            return None
        return response_text.strip() or None

    async def _stream_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        preamble: str = "",
        host_tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
        final_response_sink: Callable[[Any], None] | None = None,
    ):
        full_prompt = _merge_preamble(preamble, prompt)
        run_result = self._run_agent(
            agent_id=agent_id,
            prompt=full_prompt,
            session_id=session_id,
            stream=True,
            host_tools=host_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        if hasattr(run_result, "__aiter__") and hasattr(run_result, "get_final_response"):
            emitted = False
            async for update in run_result:
                delta = getattr(update, "text", "")
                if not delta:
                    continue
                emitted = True
                yield delta
            response = await run_result.get_final_response()
            if final_response_sink is not None:
                final_response_sink(response)
            final_text = getattr(response, "text", "")
            if final_text and not emitted:
                yield final_text
            return
        response = await run_result
        if final_response_sink is not None:
            final_response_sink(response)
        final_text = getattr(response, "text", "")
        if final_text:
            yield final_text

    def _run_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        stream: bool,
        host_tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
    ):
        agent = self._agent_builder(
            self.registry,
            agent_id,
        )
        allowed_tools, tool_options = resolve_agent_tool_policy(
            tools=tuple(host_tools),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        run_kwargs = {
            "session": agent.create_session(session_id=session_id),
            "stream": stream,
        }
        declaration_tools = build_declaration_tools(allowed_tools)
        if declaration_tools:
            run_kwargs["tools"] = declaration_tools
        run_kwargs.update(tool_options)
        result = agent.run(
            [
                Message(
                    role="user",
                    text=prompt,
                    author_name="proxy",
                )
            ],
            **run_kwargs,
        )
        if stream:
            return result
        return result

    def _emit_tool_calls(
        self,
        *,
        response: Any,
        continuation: ContinuationState,
        state: dict[str, Any],
    ) -> list[ProxyToolCallEvent]:
        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            return []
        encoded_calls = tuple(
            encode_continuation_tool_call(tool_call, state=continuation)
            for tool_call in tool_calls
        )
        state["tool_calls"] = encoded_calls
        state["finish_reason"] = "tool_calls"
        return [ProxyToolCallEvent(call) for call in encoded_calls]


def _merge_preamble(preamble: str, prompt: str) -> str:
    preamble = preamble.strip()
    prompt = prompt.strip()
    if preamble and prompt:
        return f"{preamble}\n\n{prompt}"
    return preamble or prompt


def _direct_reply_prompt(request) -> str:
    return "\n".join(
        [
            "You are responding to the host user in proxy mode.",
            "Use the full conversation transcript below as context.",
            "",
            summarize_conversation(request.messages, limit=12),
            "",
            "Latest user request:",
            request.latest_user_text() or "(none)",
        ]
    ).strip()


def _specialist_prompt(*, specialist_id: str, request_text: str, transcript_summary: str, current_brief: str | None = None) -> str:
    lines = [
        f"You are the {specialist_id} working inside the orchestration proxy.",
        "The orchestrator distilled the host conversation for you.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Assigned request:",
        request_text or "(none)",
    ]
    if current_brief:
        lines.extend(
            [
                "",
                "Current progress:",
                current_brief,
            ]
        )
    return "\n".join(lines).strip()


def _workflow_step_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    goal: str,
    current_brief: str,
    transcript_summary: str,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} working inside workflow {workflow_id}.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Overall goal:",
        goal or "(none)",
        "",
        "Current brief:",
        current_brief or "(none)",
    ]
    if prior_outputs:
        lines.extend(
            [
                "",
                "Prior workflow outputs:",
                *prior_outputs[-6:],
            ]
        )
    return "\n".join(lines).strip()


def _summary_instructions() -> str:
    return "\n".join(
        [
            "Summarize the completed work for the host user.",
            "Be concise and concrete.",
            "State what was decided or produced.",
            "Do not mention hidden chain-of-thought.",
        ]
    )


def _delegation_summary_prompt(*, request_text: str, specialist_id: str, specialist_text: str) -> str:
    return "\n".join(
        [
            f"The specialist {specialist_id} completed delegated work.",
            "",
            "Original request:",
            request_text or "(none)",
            "",
            "Specialist output:",
            specialist_text or "(none)",
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()


def _workflow_summary_prompt(*, workflow_id: str, goal: str, outputs: tuple[str, ...]) -> str:
    return "\n".join(
        [
            f"The workflow {workflow_id} completed.",
            "",
            "Goal:",
            goal or "(none)",
            "",
            "Workflow outputs:",
            *(outputs or ("(none)",)),
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()


def _set_response_holder(scope: dict[str, Any], value: Any) -> None:
    scope["response"] = value
