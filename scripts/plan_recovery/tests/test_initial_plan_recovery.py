import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.plan_recovery.evaluate_plan import Plan
from scripts.plan_recovery.utils import parse_plan, parse_actions, format_actions_to_plan
from test_utils import gen_plan, same_plan, gt_plan, subsequence, subset_plan, partial_subset, split_subsequence, initialise_chooser, initialise_evaluator_per_action

# Change directory of execution to that of the actual files
os.chdir(os.pardir)

# The result from qcode one_shot generation
inst_2_plan = parse_plan("(unstack c a)\n(put-down c)\n(unstack a b)\n(put-down a)\n(pick-up b)\n(stack b a)\n(pick-up c)\n(stack c b)\n")
inst_2_gt = parse_plan("(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)\n")

class TestEvaluatePlanParsingAndLength(unittest.TestCase):
    def test_plan_recovery(self):
        pass
    # TODO setup and test
    # def test_search_steps_to_validity_gt(self):
    #     plan = Plan(inst_2_gt, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
    #     steps = plan.steps_to_validity
    #     self.assertEqual(len(steps), 0)
    #
    # def test_search_steps_to_validity_invalid(self):
    #     plan = Plan(inst_2_plan, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
    #     steps = plan.steps_to_validity
    #     # TODO error in steps to validity, should not add actions if there are fixes (add action should be - no. of same_acts)
    #     self.assertEqual(len(steps), 8) # 2 moves, 2 fixes, 4 removals
