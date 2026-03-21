from __future__ import annotations

import unittest

from ergon_studio.proxy.response_sink import response_holder_sink


class ResponseSinkTests(unittest.TestCase):
    def test_response_holder_sink_stores_response_value(self) -> None:
        holder: dict[str, object] = {}

        sink = response_holder_sink(holder)
        sink("done")

        self.assertEqual(holder["response"], "done")
