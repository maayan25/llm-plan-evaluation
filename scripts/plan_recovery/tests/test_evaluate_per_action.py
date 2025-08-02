import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.plan_recovery.evaluate_plan_per_action import EvaluatePlanPerAction
from scripts.plan_recovery.common import get_variable_lists
from scripts.plan_recovery.utils import parse_plan, parse_actions, format_actions_to_plan
from test_utils import (gen_plan, same_plan, gt_plan, subset_plan, partial_subset, split_subsequence,
                        initialise_chooser, initialise_evaluator_per_action, gen_plan_str, same_plan_str, gt_plan_str,
                        subset_plan_str, partial_subset_str, split_subsequence_str, chooser)

chooser_same = initialise_chooser([same_plan])
chooser_gt_plan = initialise_chooser([gt_plan])
chooser_subset_plan = initialise_chooser([subset_plan])

class TestEvaluatePlanPerAction(unittest.TestCase):
    def test_evaluation_per_action_independent_from_length_same_plan(self):
        """
        Test that the evaluation of the plan is independent of the length of the plan
        """
        plan_1 = "(unstack c a)"
        chooser1 = initialise_chooser([parse_plan(plan_1)])
        diff_plan_1 = "(put-down c)"
        chooser_diff = initialise_chooser([parse_plan(diff_plan_1)])
        plan_2 = "(unstack c a)\n(unstack c a)"
        chooser2 = initialise_chooser([parse_plan(plan_2)])
        plan_3 = "(unstack c a)\n(unstack c a)\n(unstack c a)"
        chooser3 = initialise_chooser([parse_plan(plan_3)])

        evaluator_plan_length_1 = EvaluatePlanPerAction(plan_1, plan_1, chooser1, True)
        evaluator_plan_length_1.run_evaluation()

        evaluator_plan_diff = EvaluatePlanPerAction(diff_plan_1, diff_plan_1, chooser_diff, True)
        evaluator_plan_diff.run_evaluation()

        evaluator_plan_length_2 = EvaluatePlanPerAction(plan_2, plan_2, chooser2, True)
        evaluator_plan_length_2.run_evaluation()

        evaluator_plan_length_3 = EvaluatePlanPerAction(plan_3, plan_3, chooser3, True)
        evaluator_plan_length_3.run_evaluation()

        # Trace score should be the same for all plans as it is normalised by the length of the plan
        self.assertEqual(evaluator_plan_length_1.curr_plan.get_trace_score(), evaluator_plan_diff.curr_plan.get_trace_score())
        self.assertEqual(evaluator_plan_length_1.curr_plan.get_trace_score(), evaluator_plan_length_2.curr_plan.get_trace_score())
        self.assertEqual(evaluator_plan_length_1.curr_plan.get_trace_score(), evaluator_plan_length_3.curr_plan.get_trace_score())

        # print(f"Similarity scores: {evaluator_plan_length_1.curr_plan.get_similarity_score()}, {evaluator_plan_diff.curr_plan.get_similarity_score()}, {evaluator_plan_length_2.curr_plan.get_similarity_score()}, {evaluator_plan_length_3.curr_plan.get_similarity_score()}")
        # print(f"Final scores: {evaluator_plan_length_1.final_score}, {evaluator_plan_diff.final_score}, {evaluator_plan_length_2.final_score}, {evaluator_plan_length_3.final_score}")
        self.assertEqual(evaluator_plan_length_1.final_score, evaluator_plan_diff.final_score)
        self.assertLessEqual(evaluator_plan_length_1.final_score, evaluator_plan_length_2.final_score)
        self.assertLessEqual(evaluator_plan_length_1.final_score, evaluator_plan_length_3.final_score)
        self.assertLessEqual(evaluator_plan_length_2.final_score, evaluator_plan_length_3.final_score)

    def test_evaluation_per_action_independent_from_length_gt_plan(self):
        """
        Test that the evaluation of the plan is independent of the length of the plan
        """
        plan_1 = "(unstack c a)"
        plan_2 = "(unstack c a)\n(unstack c a)"
        plan_3 = "(unstack c a)\n(unstack c a)\n(unstack c a)"

        gt_plan_str_len_2 = "(unstack c a)\n(put-down c)"
        chooser_len_2 = initialise_chooser([parse_plan(gt_plan_str_len_2)])

        evaluator_plan_length_1 = initialise_evaluator_per_action(plan_1, gt_plan_str_len_2, chooser_len_2)
        evaluator_plan_length_2 = initialise_evaluator_per_action(plan_2, gt_plan_str_len_2, chooser_len_2)
        evaluator_plan_length_3 = initialise_evaluator_per_action(plan_3, gt_plan_str_len_2, chooser_len_2)

        # An action can be changed, no action needs to be added TODO not sure about this case
        self.assertGreater(evaluator_plan_length_2.curr_plan.trace_score, evaluator_plan_length_1.curr_plan.trace_score)
        self.assertGreater(evaluator_plan_length_1.curr_plan.get_score(), evaluator_plan_length_2.curr_plan.get_score())

        # Length 2 is closer to the GT plan than length 3
        # Trace score is the same because redundant action is removed, but the score is higher for the shorter plan
        self.assertEqual(evaluator_plan_length_2.curr_plan.trace_score, evaluator_plan_length_3.curr_plan.trace_score)
        self.assertGreater(evaluator_plan_length_2.get_final_score(), evaluator_plan_length_3.get_final_score())

    def test_evaluation_per_action_higher_than_invalid(self):
        """
        Test that the evaluation of the plan is higher for short valid plans than for long invalid plans
        """
        plan_1 = "(unstack c a)"
        chooser1 = initialise_chooser([parse_plan(plan_1)])
        diff_plan_1 = "(put-down c)"
        plan_2 = "(unstack c a)\n(unstack c a)"
        chooser2 = initialise_chooser([parse_plan(plan_2)])
        diff_plan_2 = "(put-down c)"
        very_long_plan = "(unstack c a)\n(unstack c a)\n(unstack c a)\n(unstack c a)\n(unstack c a)\n(unstack c a)"
        long_gt = "(unstack c a)\n(put-down c)\n(unstack a b)\n(put-down a)\n(pick-up b)\n(stack b a)\n(pick-up c)\n(stack c b)"
        chooser_long = initialise_chooser([parse_plan(long_gt)])

        # Valid plan of length 1
        evaluator_plan_length_1 = EvaluatePlanPerAction(plan_1, plan_1, chooser1, True)
        evaluator_plan_length_1.run_evaluation()

        # Invalid plan of length 1
        evaluator_plan_diff_1 = initialise_evaluator_per_action(diff_plan_1, diff_plan_1, chooser1)

        # Valid plan of length 2
        evaluator_plan_length_2 = EvaluatePlanPerAction(plan_2, plan_2, chooser2, True)
        evaluator_plan_length_2.run_evaluation()

        # Invalid plans of length 2
        evaluator_plan_diff_2 = initialise_evaluator_per_action(diff_plan_2, diff_plan_2, chooser2)
        evaluator_plan_diff_2_1 = initialise_evaluator_per_action(diff_plan_2, diff_plan_2, chooser1)

        # Invalid plan of length 6
        evaluator_very_long_plan_gt_1 = initialise_evaluator_per_action(very_long_plan, very_long_plan, chooser1)
        evaluator_very_long_plan_gt_2 = initialise_evaluator_per_action(very_long_plan, very_long_plan, chooser2)
        evaluator_very_long_plan_gt_long = initialise_evaluator_per_action(very_long_plan, long_gt, chooser_long)

        # Valid plan should have higher score than invalid plan
        self.assertGreater(evaluator_plan_length_1.final_score, evaluator_plan_diff_1.final_score)
        self.assertGreater(evaluator_plan_length_2.final_score, evaluator_plan_diff_2.final_score)

        # Valid plan of length 1 should have higher score than invalid plan of length 2
        self.assertGreater(evaluator_plan_length_1.final_score, evaluator_plan_diff_2.final_score)
        self.assertGreater(evaluator_plan_length_1.final_score, evaluator_plan_diff_2_1.final_score)
        self.assertGreater(evaluator_plan_length_2.final_score, evaluator_plan_diff_1.final_score)

        # Valid plan of length 2 should have higher score than invalid plan of length 6
        self.assertGreater(evaluator_plan_length_2.final_score, evaluator_very_long_plan_gt_1.final_score)
        self.assertGreater(evaluator_plan_length_2.final_score, evaluator_very_long_plan_gt_2.final_score)

        # Valid plan of length 1 should have higher score than invalid plan of length 6
        self.assertGreater(evaluator_plan_length_1.final_score, evaluator_very_long_plan_gt_1.final_score)
        self.assertGreater(evaluator_plan_length_1.final_score, evaluator_very_long_plan_gt_2.final_score)
        self.assertGreater(evaluator_plan_length_1.final_score, evaluator_very_long_plan_gt_long.final_score)


    def test_parse_plan(self):
        parsed_plan = parse_plan(gen_plan_str)
        self.assertEqual(parsed_plan, gen_plan)
        parsed_plan = parse_plan(gt_plan_str)
        self.assertEqual(parsed_plan, gt_plan)
        parsed_plan = parse_plan(partial_subset_str)
        self.assertEqual(parsed_plan, partial_subset)
        parsed_plan = parse_plan(split_subsequence_str)
        self.assertEqual(parsed_plan, split_subsequence)

    def test_parse_plans_and_actions(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser_same)
        self.assertEqual(len(evaluator.gt_plan), 6)
        self.assertEqual(len(evaluator.gt_actions), 6)

    def test_parse_plans_3_params(self):
        plan: str = "(load-airplane p0 a0 l1-0)\n(load-airplane p1 a0 l1-0)\n(fly-airplane a0 l1-0 l0-0)\n(unload-airplane p0 a0 l0-0)\n(unload-airplane p1 a0 l0-0)\n"
        parsed_plan = parse_plan(plan)
        self.assertEqual(len(parsed_plan), 5)
        parsed_actions = parse_actions(parsed_plan)
        self.assertEqual(len(parsed_actions), 5)
        self.assertEqual(parsed_actions[0], ["load-airplane", "p0", "a0", "l1-0"])

    def test_format_actions_to_plans(self):
        actions = parse_actions(gen_plan)
        plan = format_actions_to_plan(actions)
        self.assertEqual(gen_plan, plan)

    """
    Test the penalise_length_from_chosen and penalise_length_from_optimal methods
    """

    def test_penalise_length_from_chosen_same_plan(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser_same)
        length_score = evaluator.penalise_length_from_chosen(gen_plan)
        self.assertEqual(length_score, 0.0)

    def test_penalise_length_from_chosen_longer_curr_plan(self):
        # subset_plan is shorter than gen_plan
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser_subset_plan)
        length_score = evaluator.penalise_length_from_chosen(gen_plan)
        length_diff = len(gen_plan) - len(subset_plan)
        self.assertEqual(length_score, length_diff ** 2 / len(subset_plan))
        self.assertFalse(evaluator.generation_required)

    def test_penalise_length_from_chosen_shorter_curr_plan(self):
        # gt_plan is longer than gen_plan
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser_gt_plan)
        length_score = evaluator.penalise_length_from_chosen(gen_plan)
        length_diff = len(gen_plan) - len(gt_plan)
        self.assertEqual(length_score, 2 * (length_diff ** 2) / len(gt_plan))
        self.assertTrue(evaluator.generation_required)

    def test_penalise_length_from_opt_same_plan(self):
        # Chosen plan is longer, but optimal plan is the same
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser_gt_plan)
        length_score = evaluator.penalise_length_from_optimal(same_plan)
        self.assertEqual(length_score, 0.0)

    def test_penalise_length_from_opt_longer_curr_plan(self):
        # Chosen plan is same, but optimal plan is longer (not actually possible... but chosen should not matter)
        evaluator = EvaluatePlanPerAction(gen_plan_str, subset_plan_str, chooser)
        length_score = evaluator.penalise_length_from_optimal(gen_plan)
        self.assertEqual(length_score, (len(gen_plan) - len(subset_plan)) / len(subset_plan))

    def test_penalise_length_from_opt_shorter_curr_plan(self):
        # Chosen plan is same, but optimal plan is shorter
        evaluator = EvaluatePlanPerAction(gen_plan_str, gt_plan_str, chooser)
        length_score = evaluator.penalise_length_from_optimal(gen_plan)
        self.assertEqual(length_score, -2 * (len(gen_plan) - len(gt_plan)) / len(gt_plan))

class TestEvaluatePlanPerActionQualityConsistentVariableSwap(unittest.TestCase):
    """
    Tests for scoring of consistent variable swap in variable mapping of plan
    """

    def test_get_variable_lists_1_variable(self):
        actions = [["stack", "a"]]
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(actions)

        self.assertEqual(len(variables), 1)
        self.assertIn("a", variables)

    def test_get_variable_lists_gen_plan(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(evaluator.curr_plan.actions)

        self.assertEqual(len(variables), 3)
        self.assertIn("a", variables)
        self.assertIn("b", variables)
        self.assertIn("c", variables)

    def test_get_variable_lists_many_variable(self):
        actions = [['unstack', 'a', 'b'], ['put-down', 'c'], ['pick-up', 'd'], ['stack','e', 'f'], ['pick-up', 'g)']]
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(actions)

        self.assertEqual(len(variables), 7)

    def test_count_swaps_2_vars(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "a", "c": "c"}
        swaps = evaluator.count_swaps(mapping)
        self.assertEqual(swaps, 2)

    def test_count_swaps_circular_swap(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "c", "c": "a"}
        swaps = evaluator.count_swaps(mapping)
        self.assertEqual(swaps, 3)

    def test_swap_variables_2_vars(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "a", "c": "c"}
        new_plan = evaluator.swap_variables(mapping)
        self.assertTrue(len(gen_plan) == len(new_plan))
        self.assertEqual(new_plan, [['unstack', 'c', 'b'], ['put-down', 'c'], ['pick-up', 'b'], ['stack', 'b', 'c'], ['pick-up', 'a'], ['stack', 'a', 'b']])

    def test_swap_variables_circular_swap(self):
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "c", "c": "a"}
        new_plan = evaluator.swap_variables(mapping)
        self.assertTrue(len(gen_plan) == len(new_plan))
        self.assertEqual(new_plan, [['unstack', 'a', 'b'], ['put-down', 'a'], ['pick-up', 'b'], ['stack', 'b', 'a'], ['pick-up', 'c'], ['stack', 'c', 'b']])

    def test_search_optimal_internal_swap_same_plan(self):
        # Optimal swap should be no swap because they are identical
        evaluator = EvaluatePlanPerAction(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(evaluator.curr_plan.actions)
        optimality = evaluator.search_optimal_internal_swap(variables)
        optimal_plan = optimality.get("optimal_plan")

        self.assertEqual(optimal_plan.actions, evaluator.curr_plan.actions)

    def test_search_optimal_internal_swap_gt_plan(self):
        # Optimal swap should be no swap
        evaluator = EvaluatePlanPerAction(gen_plan_str, gt_plan_str, chooser_gt_plan)
        variables = get_variable_lists(evaluator.curr_plan.actions)
        optimality = evaluator.search_optimal_internal_swap(variables)
        optimal_plan = optimality.get("optimal_plan")

        self.assertEqual(optimal_plan.actions, [['unstack', 'a', 'b'], ['put-down', 'a'], ['pick-up', 'b'], ['stack', 'b', 'a'], ['pick-up', 'c'], ['stack', 'c', 'b']])

    def test_search_optimal_internal_swap_swapped_plan(self):
        # Optimal swap should be actions of OG gen_plan
        swapped_gen_plan_str = "(unstack c b)\n(put-down c)\n(pick-up b)\n(stack b c)\n(pick-up a)\n(stack a b)"
        evaluator = EvaluatePlanPerAction(swapped_gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(evaluator.curr_plan.actions)
        optimality = evaluator.search_optimal_internal_swap(variables)
        optimal_plan = optimality.get("optimal_plan")

        # The OG actions of gen_plan before the swap!
        self.assertEqual(optimal_plan.actions, [['unstack', 'c', 'a'], ['put-down', 'c'], ['pick-up', 'a'], ['stack', 'a', 'c'], ['pick-up', 'b'], ['stack', 'b', 'a']])


    def test_consistent_variable_swap(self):
        pass

if __name__ == '__main__':
    unittest.main()