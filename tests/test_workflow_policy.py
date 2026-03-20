from __future__ import annotations

import unittest

from ergon_studio.workflow_policy import acceptance_mode_for_metadata, delivery_candidate_for_metadata, selection_hints_for_metadata


class WorkflowPolicyTests(unittest.TestCase):
    def test_acceptance_mode_defaults_to_delivery(self) -> None:
        self.assertEqual(acceptance_mode_for_metadata({}), "delivery")

    def test_delivery_candidate_follows_metadata(self) -> None:
        self.assertTrue(delivery_candidate_for_metadata({"delivery_candidate": True}))
        self.assertTrue(delivery_candidate_for_metadata({"acceptance_mode": "delivery"}))
        self.assertFalse(delivery_candidate_for_metadata({"acceptance_mode": "design_brief"}))

    def test_selection_hints_for_metadata_normalizes_and_deduplicates(self) -> None:
        self.assertEqual(
            selection_hints_for_metadata({"selection_hints": ["tiny_delivery", " tiny_delivery ", "", "adaptive_delivery"]}),
            ("tiny_delivery", "adaptive_delivery"),
        )
        self.assertEqual(selection_hints_for_metadata({"selection_hints": "tiny_delivery"}), ())
