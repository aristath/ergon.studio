from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelMessage:
    author: str
    content: str

    def render(self) -> str:
        return f"{self.author}: {self.content}"


@dataclass
class OpenChannel:
    channel_id: str
    name: str
    participants: tuple[str, ...]
    transcript: list[ChannelMessage] = field(default_factory=list)


@dataclass
class ChannelSession:
    session_id: str
    channels: dict[str, OpenChannel] = field(default_factory=dict)


class ChannelStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChannelSession] = {}

    def get(self, session_id: str) -> ChannelSession | None:
        return self._sessions.get(session_id)

    def ensure(self, session_id: str) -> ChannelSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = ChannelSession(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def put(self, session: ChannelSession) -> None:
        self._sessions[session.session_id] = session

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def describe_open_channels(
    channels: dict[str, OpenChannel],
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
