from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
_VALID_FINISH_REASONS = {"stop", "tool_calls", "length", "content_filter", "error"}


@dataclass(frozen=True)
class ProxyToolCall:
    id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tool call id must be non-empty")
        if not self.name:
            raise ValueError("tool call name must be non-empty")
        if not isinstance(self.arguments_json, str):
            raise TypeError("arguments_json must be a string")


@dataclass(frozen=True)
class ProxyFunctionTool:
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must be non-empty")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dict")
        if type(self.strict) is not bool:
            raise TypeError("strict must be a bool")


@dataclass(frozen=True)
class ProxyInputMessage:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ProxyToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in _VALID_MESSAGE_ROLES:
            raise ValueError(f"unsupported message role: {self.role}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if self.name is not None and not self.name:
            raise ValueError("name must be non-empty when provided")
        if self.tool_call_id is not None and not self.tool_call_id:
            raise ValueError("tool_call_id must be non-empty when provided")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may include tool_calls")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool messages may include tool_call_id")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple")


@dataclass(frozen=True)
class ProxyTurnRequest:
    model: str
    messages: tuple[ProxyInputMessage, ...]
    tools: tuple[ProxyFunctionTool, ...] = ()
    stream: bool = False
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple")
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")
        if type(self.stream) is not bool:
            raise TypeError("stream must be a bool")
        if self.parallel_tool_calls is not None and type(self.parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be a bool or None")

    def latest_user_message(self) -> ProxyInputMessage | None:
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None

    def latest_user_text(self) -> str | None:
        message = self.latest_user_message()
        if message is None:
            return None
        return message.content


@dataclass(frozen=True)
class ProxyReasoningDeltaEvent:
    delta: str

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str):
            raise TypeError("delta must be a string")


@dataclass(frozen=True)
class ProxyContentDeltaEvent:
    delta: str

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str):
            raise TypeError("delta must be a string")


@dataclass(frozen=True)
class ProxyToolCallEvent:
    call: ProxyToolCall

    def __post_init__(self) -> None:
        if not isinstance(self.call, ProxyToolCall):
            raise TypeError("call must be a ProxyToolCall")


@dataclass(frozen=True)
class ProxyFinishEvent:
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in _VALID_FINISH_REASONS:
            raise ValueError(f"unsupported finish reason: {self.reason}")
