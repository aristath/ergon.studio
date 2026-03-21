from __future__ import annotations

import unittest

from ergon_studio.workroom_policy import (
    workroom_acceptance_mode_for_metadata,
    workroom_delivery_candidate_for_metadata,
    workroom_selection_hints_for_metadata,
)


class WorkroomPolicyTests(unittest.TestCase):
    def test_acceptance_mode_defaults_to_delivery(self) -> None:
        self.assertEqual(workroom_acceptance_mode_for_metadata({}), "delivery")

    def test_delivery_candidate_follows_metadata(self) -> None:
        self.assertTrue(
            workroom_delivery_candidate_for_metadata({"delivery_candidate": True})
        )
        self.assertTrue(
            workroom_delivery_candidate_for_metadata({"acceptance_mode": "delivery"})
        )
        self.assertFalse(
            workroom_delivery_candidate_for_metadata(
                {"acceptance_mode": "design_brief"}
            )
        )

    def test_workroom_selection_hints_for_metadata_normalizes_and_deduplicates(
        self,
    ) -> None:
        self.assertEqual(
            workroom_selection_hints_for_metadata(
                {
                    "selection_hints": [
                        "tiny_delivery",
                        " tiny_delivery ",
                        "",
                        "adaptive_delivery",
                    ]
                }
            ),
            ("tiny_delivery", "adaptive_delivery"),
        )
        self.assertEqual(
            workroom_selection_hints_for_metadata(
                {"selection_hints": "tiny_delivery"}
            ),
            (),
        )
