from __future__ import annotations

import unittest

from ergon_studio.workflow_policy import acceptance_criteria_for_mode, acceptance_mode_for_metadata, acceptance_rule_for_mode, is_decision_ready_acceptance_mode, is_non_delivery_acceptance_mode, is_planning_acceptance_mode


class WorkflowPolicyTests(unittest.TestCase):
    def test_acceptance_mode_defaults_to_delivery(self) -> None:
        self.assertEqual(acceptance_mode_for_metadata({}), "delivery")

    def test_acceptance_mode_helpers_classify_modes(self) -> None:
        self.assertTrue(is_non_delivery_acceptance_mode("design_brief"))
        self.assertFalse(is_non_delivery_acceptance_mode("delivery"))
        self.assertTrue(is_decision_ready_acceptance_mode("decision_ready"))
        self.assertFalse(is_decision_ready_acceptance_mode("delivery"))
        self.assertTrue(is_planning_acceptance_mode("research_brief"))
        self.assertFalse(is_planning_acceptance_mode("delivery"))

    def test_acceptance_rule_and_criteria_follow_mode(self) -> None:
        self.assertIn("decision-ready recommendation", acceptance_rule_for_mode("decision_ready"))
        self.assertIn("research brief", acceptance_rule_for_mode("research_brief"))
        self.assertIn("minimal working delivery", acceptance_rule_for_mode("delivery"))
        self.assertIn("decision-ready recommendation", acceptance_criteria_for_mode("decision_ready"))
        self.assertIn("minimal working result", acceptance_criteria_for_mode("delivery"))
