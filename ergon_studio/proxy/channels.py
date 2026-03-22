from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelMessage:
    author: str
    content: str

    def render(self) -> str:
        return f"{self.author}: {self.content}"


@dataclass
class Channel:
    channel_id: str
    name: str
    participants: tuple[str, ...]
    transcript: list[ChannelMessage] = field(default_factory=list)


class ChannelStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Channel]] = {}

    def get(self, session_id: str) -> dict[str, Channel] | None:
        return self._sessions.get(session_id)

    def put(self, session_id: str, channels: dict[str, Channel]) -> None:
        self._sessions[session_id] = channels

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def describe_open_channels(
    channels: dict[str, Channel],
) -> tuple[str, ...]:
    descriptions: list[str] = []
    for channel_id, channel in sorted(channels.items()):
        roster = ", ".join(channel.participants) or "(none)"
        if channel.transcript:
            recent = " | ".join(
                message.render() for message in channel.transcript[-3:]
            )
            descriptions.append(f"{channel_id}: {channel.name} [{roster}] :: {recent}")
        else:
            descriptions.append(f"{channel_id}: {channel.name} [{roster}]")
    return tuple(descriptions)
