from __future__ import annotations

import unittest

from ergon_studio.proxy.channels import Channel, ChannelMessage, describe_open_channels


class DescribeOpenChannelsTests(unittest.TestCase):
    def test_shows_truncation_indicator_when_transcript_exceeds_three(self) -> None:
        channel = Channel(
            channel_id="channel-1",
            name="code-review",
            participants=("coder", "reviewer"),
            transcript=[
                ChannelMessage(author="orchestrator", content=f"msg {i}")
                for i in range(6)
            ],
        )
        descriptions = describe_open_channels({"channel-1": channel})
        self.assertEqual(len(descriptions), 1)
        self.assertIn("[3 earlier]", descriptions[0])

    def test_truncation_indicator_shows_correct_count(self) -> None:
        channel = Channel(
            channel_id="channel-1",
            name="code-review",
            participants=("coder",),
            transcript=[
                ChannelMessage(author="coder", content=f"msg {i}")
                for i in range(10)
            ],
        )
        descriptions = describe_open_channels({"channel-1": channel})
        self.assertIn("[7 earlier]", descriptions[0])

    def test_no_truncation_indicator_for_three_messages(self) -> None:
        channel = Channel(
            channel_id="channel-1",
            name="code-review",
            participants=("coder",),
            transcript=[
                ChannelMessage(author="coder", content=f"msg {i}")
                for i in range(3)
            ],
        )
        descriptions = describe_open_channels({"channel-1": channel})
        self.assertNotIn("earlier", descriptions[0])

    def test_no_truncation_indicator_for_empty_transcript(self) -> None:
        channel = Channel(
            channel_id="channel-1",
            name="debug",
            participants=("coder",),
        )
        descriptions = describe_open_channels({"channel-1": channel})
        self.assertEqual(len(descriptions), 1)
        self.assertNotIn("earlier", descriptions[0])
