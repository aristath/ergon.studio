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
