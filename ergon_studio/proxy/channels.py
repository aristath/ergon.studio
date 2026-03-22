from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelSnapshot:
    channel_id: str
    name: str
    participants: tuple[str, ...]
    transcript: tuple[str, ...] = ()


@dataclass
class OpenChannel:
    channel_id: str
    name: str
    participants: tuple[str, ...]
    transcript: list[str] = field(default_factory=list)

    def snapshot(self) -> ChannelSnapshot:
        return ChannelSnapshot(
            channel_id=self.channel_id,
            name=self.name,
            participants=self.participants,
            transcript=tuple(self.transcript),
        )


def describe_open_channels(
    channels: dict[str, OpenChannel],
) -> tuple[str, ...]:
    descriptions: list[str] = []
    for channel_id, channel in sorted(channels.items()):
        roster = ", ".join(channel.participants) or "(none)"
        if channel.transcript:
            recent = " | ".join(channel.transcript[-3:])
            descriptions.append(f"{channel_id}: {channel.name} [{roster}] :: {recent}")
        else:
            descriptions.append(f"{channel_id}: {channel.name} [{roster}]")
    return tuple(descriptions)
