import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.plan_recovery.evaluate_plan import Plan
from scripts.plan_recovery.utils import parse_plan
from test_utils import gen_plan, same_plan, gt_plan, subsequence, subset_plan, initialise_chooser

# Change directory of execution to that of the actual files
os.chdir(os.pardir)

# The result from qcode one_shot generation
inst_2_plan = parse_plan("(unstack c a)\n(put-down c)\n(unstack a b)\n(put-down a)\n(pick-up b)\n(stack b a)\n(pick-up c)\n(stack c b)\n")
inst_2_gt = parse_plan("(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)\n")

class TestPlanActionCorrectnessAndPositioning(unittest.TestCase):
    """
    Test functions for action similarity score (positioning and correctness)
    """
    def test_check_identical_actions_gt_plan(self):
        gt = Plan(inst_2_gt, inst_2_gt, True)
        curr_actions, gt_actions, positioning_score = gt.check_action_positioning()
        self.assertEqual(positioning_score, len(inst_2_gt))
        _, _, correctness_score = gt.check_action_correctness(curr_actions, gt_actions)
        self.assertEqual(0, correctness_score) # no actions should be in wrong position

    def test_check_identical_actions_subset_plan(self):
        # Test with a plan (subset_plan) which exists as is in gen_plan, but in the middle
        subset = Plan(subset_plan, gen_plan, True)  # subset_plan is a pure subset of gen_plan, but positioned in the middle
        curr_actions, gt_actions, positioning_score = subset.check_action_positioning()
        self.assertEqual(positioning_score, 0) # all actions are in wrong position
        _, _, correctness_score = subset.check_action_correctness(curr_actions, gt_actions)
        self.assertEqual(correctness_score, 0.5 * len(subset_plan)) # all actions are in wrong position

    def test_check_identical_actions_subsequence(self):
        # Test with a plan where all gt_actions are in curr_actions, but some are in wrong positions
        subseq = Plan(subsequence, gen_plan, True)  # subsequence is a subsequence of gen_plan
        curr_actions, gt_actions, positioning_score = subseq.check_action_positioning()
        _, _, correctness_score = subseq.check_action_correctness(curr_actions, gt_actions)

        # 2 first actions are in right position, 3 last actions are in offset of 1
        corect_pos = 2
        self.assertEqual(positioning_score, corect_pos)
        self.assertEqual(correctness_score, 0.5 * (len(subsequence) - corect_pos))

    def test_check_identical_actions_different_plan(self):
        # Test with a different plan (gt_plan) which is not a specific subset of gen_plan
        gen = Plan(gen_plan, gt_plan, True)
        curr_actions, gt_actions, positioning_score = gen.check_action_positioning()
        _, _, correctness_score = gen.check_action_correctness(curr_actions, gt_actions)

        # No actions in gen_plan are correct in the wrong position
        self.assertEqual(positioning_score, 4)
        self.assertEqual(correctness_score, 0.5 * 0)

plan = Plan(gen_plan, gt_plan, True)

class TestEvaluatePlanQualityActionSimilarity(unittest.TestCase):
    """
    Test comparison of variables between 2 actions
    """
    def test_compare_variables_binary_action_dissimilar(self):
        # Plan with same action, but different variables
        similarity = plan.compare_variables(["stack", "a", "b"], ["stack", "c", "d"])
        self.assertEqual(similarity, 0)

    def test_compare_variables_binary_action_swap_1(self):
        # Plan with same action, but 1 var is swapped and 1 is different
        similarity = plan.compare_variables(["stack", "a", "b"], ["stack", "c", "a"])
        self.assertEqual(similarity, 0.25)

    def test_compare_variables_binary_action_1_correct_var(self):
        # Plan with same action, 1 var is correct and 1 is different
        similarity = plan.compare_variables(["stack", "a", "b"], ["stack", "a", "c"])
        self.assertEqual(similarity, 0.5)

    def test_compare_variables_binary_action_swapped_vars(self):
        # Plan with same action, but variables are swapped
        similarity = plan.compare_variables(["stack", "a", "b"], ["stack", "b", "a"])
        self.assertEqual(similarity, 0.75)

    def test_compare_variables_binary_action_correct_vars(self):
        similarity = plan.compare_variables(["stack", "a", "b"], ["unstack", "a", "b"])
        self.assertEqual(similarity, 1)

    def test_compare_variables_binary_action_gt_is_unary_incorrect_vars(self):
        similarity = plan.compare_variables(["load", "a", "b"], ["load", "c"])
        self.assertEqual(similarity, -0.1)

    def test_compare_variables_binary_action_gt_is_unary_correct_var_1(self):
        similarity = plan.compare_variables(["load", "a", "b"], ["load", "a"])
        self.assertEqual(similarity, 0.4)

    def test_compare_variables_binary_action_gt_is_unary_correct_var_2(self):
        similarity = plan.compare_variables(["load", "a", "b"], ["load", "b"])
        self.assertEqual(similarity, 0.4)

    def test_compare_variables_binary_action_gt_is_unary_correct_both(self):
        similarity = plan.compare_variables(["load", "b", "b"], ["load", "b"])
        self.assertEqual(similarity, 0.65)

    def test_compare_variables_unary_action_incorrect_var(self):
        similarity = plan.compare_variables(["load", "a"], ["load", "b"])
        self.assertEqual(similarity, 0)

    def test_compare_variables_unary_action_correct_var(self):
        similarity = plan.compare_variables(["load", "a"], ["load", "a"])
        self.assertEqual(similarity, 0.75)

    def test_compare_variables_unary_action_gt_is_binary_incorrect_var(self):
        similarity = plan.compare_variables(["stack", "a"], ["stack", "c", "b"])
        self.assertEqual(similarity, -0.2)

    def test_compare_variables_unary_action_gt_is_binary_correct_var_1(self):
        similarity = plan.compare_variables(["stack", "a"], ["stack", "a", "b"])
        self.assertEqual(similarity, 0.3)

    def test_compare_variables_unary_action_gt_is_binary_correct_var_2(self):
        similarity = plan.compare_variables(["stack", "a"], ["stack", "b", "a"])
        self.assertEqual(similarity, 0.3)


    """
    Test functions for action similarity functions (cannot test the actual similarity score, was tested separately)
    """
    def test_get_action_pairs_gt(self):
        # curr_actions and gt_actions are the same and empty
        action_similarity = plan.get_action_pairs([], [])
        self.assertEqual(action_similarity, [])

    def test_get_action_pairs_same_act_some_vars(self):
        gen = Plan(['(pick-up a)', '(stack a c)'], ['(pick-up c)', '(stack a b)'], True)
        print(gen.actions)
        print(gen.action_pairs)
        self.assertEqual(len(gen.action_pairs), 2)
        self.assertEqual(gen.action_pairs[0], (['pick-up', 'a'], ['pick-up', 'c'], "same_act"))
        self.assertEqual(gen.action_pairs[1], (['stack', 'a', 'c'], ['stack', 'a', 'b'], "same_act"))

    def test_get_action_pairs_same_act_diff_vars(self):
        gen = Plan(['(pick-up a)', '(stack c d)', '(load b)'], ['(pick-up c)', '(stack a b)', '(load a)'], True)
        print(gen.actions)
        print(gen.action_pairs)
        self.assertEqual(len(gen.action_pairs), 3)
        # self.assertEqual(gen.action_pairs[0], (['pick-up', 'a'], ['pick-up', 'c'], "same_act"))
        self.assertEqual(gen.action_pairs[1], (['stack', 'c', 'd'], ['stack', 'a', 'b'], "same_act"))
        self.assertEqual(gen.action_pairs[2], (['load', 'b'], ['load', 'a'], "same_act"))

    def test_get_action_pairs_diff_act_same_vars(self):
        gen = Plan(['(pick c)', '(unstack a b)'], ['(pick-up c)', '(stack a b)'], True)
        print(gen.actions)
        print(gen.action_pairs)
        self.assertEqual(len(gen.action_pairs), 2)
        self.assertEqual(gen.action_pairs[0], (['pick', 'c'], ['pick-up', 'c'], "diff_act"))
        self.assertEqual(gen.action_pairs[1], (['unstack', 'a', 'b'], ['stack', 'a', 'b'], "diff_act"))


class TestEvaluatePlanTrace(unittest.TestCase):
    def test_plan_trace_same(self):
        """
        Test that plan trace works as expected
        """
        plan = Plan(gen_plan, same_plan, True, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        self.assertEqual(len(plan_trace), len(gen_plan))
        self.assertIn("position", plan_trace[0])
        self.assertNotIn("correct", plan_trace[1])

    def test_check_action_similarity_pure_subset_plan(self):
        """
        Test with a plan (subset_plan) which exists as is in gen_plan, but in the middle
        """
        subset = Plan(subset_plan, gen_plan,True)

        # Assert plan trace only contains "correct" keys
        plan_trace = subset.plan_trace

        self.assertEqual(len(plan_trace), len(subset_plan))
        self.assertNotIn("position", plan_trace.values())
        self.assertIn("correct", plan_trace.values())
        self.assertNotIn("same_act", plan_trace.values())
        self.assertNotIn("diff_act", plan_trace.values())
        self.assertNotIn("redundant", plan_trace.values())

        # Should be 0 because nothing to compare to
        print(subset.plan_trace)
        self.assertEqual(subset.get_trace_score(), (2 * 4) / len(subset_plan)) # each correct action is 2 points, normalised by length of the generated plan

    def test_plan_trace_gen(self):
        """
        Test that plan trace works as expected
        """
        plan = Plan(gen_plan, gt_plan, True, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        self.assertEqual(len(plan_trace), len(gen_plan))

    def test_plan_trace_else(self):
        """
        Test that plan trace works as expected
        """
        gen = ['(unstack a b)', '(put-down a)', '(unstack b c)', '(unstack a b)', '(put-down a)', '(unstack b c)', '(stack b a)']
        gt = ['(unstack a b)', '(put-down a)', '(unstack b c)', '(stack b a)']
        plan = Plan(gen, gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        self.assertEqual(len(plan_trace), len(gen))

    def test_check_validity_gt_plan(self):
        plan = Plan(inst_2_gt, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
        # plan should be valid
        self.assertTrue(plan.is_valid)

    def test_check_validity_invalid_plan(self):
        plan = Plan(inst_2_plan, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
        # plan should be invalid
        self.assertFalse(plan.is_valid)

    def test_plan_trace_gt(self):
        plan = Plan(inst_2_gt, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        self.assertEqual(len(plan_trace), len(inst_2_gt))
        self.assertEqual(plan_trace, {0: 'position', 1: 'position', 2: 'position', 3: 'position'})

    def test_plan_trace_invalid(self):
        plan = Plan(inst_2_plan, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        self.assertEqual(len(plan_trace), len(inst_2_plan))
        self.assertEqual(plan_trace, {4: 'correct', 5: 'correct', 0: 'diff_act', 1: 'redundant', 2: 'diff_act', 3: 'redundant', 6: 'redundant', 7: 'redundant'})

    def test_plan_trace_no_redundant(self):
        gen = ['(unstack a b)', '(put-down c)', '(unstack a b)', '(put-down c)']
        gt = ['(unstack c a)', '(put-down c)', '(unstack a b)', '(put-down a)', '(pick-up c)', '(stack c b)', '(pick-up a)', '(stack a c)']
        plan = Plan(gen, gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        print(plan_trace)
        self.assertEqual(len(plan_trace), len(gen))
        self.assertEqual(plan_trace, {1: 'position', 2: 'position', 0: 'same_act', 3: 'same_act'})

    def test_nr_plan_trace_gt(self):
        plan = Plan(inst_2_gt, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        nr_plan_trace = plan.nr_plan_trace
        self.assertEqual(len(plan_trace), len(inst_2_gt))
        self.assertEqual(plan_trace, {0: 'position', 1: 'position', 2: 'position', 3: 'position'})
        self.assertEqual(plan_trace, nr_plan_trace)

    def test_nr_plan_trace_invalid(self):
        plan = Plan(inst_2_plan, inst_2_gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        nr_plan_trace = plan.nr_plan_trace
        self.assertEqual(len(plan_trace), len(inst_2_plan))
        self.assertEqual(plan_trace, {4: 'correct', 5: 'correct', 0: 'diff_act', 1: 'redundant', 2: 'diff_act', 3: 'redundant', 6: 'redundant', 7: 'redundant'})
        self.assertEqual(nr_plan_trace, {4: 'correct', 5: 'correct', 0: 'diff_act', 2: 'diff_act', 3: 'diff_act', 6: 'same_act', 7: 'same_act', 1: 'diff_act'})

    def test_nr_plan_trace_redundant(self):
        gen = ['(unstack a b)', '(put-down c)', '(unstack a b)', '(put-down c)']
        gt = ['(unstack c a)', '(put-down c)', '(unstack a b)', '(put-down a)', '(pick-up c)', '(stack c b)', '(pick-up a)', '(stack a c)']
        plan = Plan(gen, gt, False, domain="blocksworld_3", instance_id=2)
        plan_trace = plan.plan_trace
        print(plan_trace)
        self.assertEqual(len(plan_trace), len(gen))
        self.assertEqual(plan_trace, {1: 'position', 2: 'position', 0: 'same_act', 3: 'same_act'})


if __name__ == '__main__':
    unittest.main()

