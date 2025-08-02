import unittest

from scripts.plan_recovery.evaluate_plan_quality import EvaluatePlan
from scripts.plan_recovery.utils import parse_plan, parse_actions, format_actions_to_plan
from scripts.tests.test_utils import chooser
from test_utils import gen_plan, same_plan, gt_plan, subsequence, subset_plan, partial_subset, split_subsequence, initialise_chooser, initialise_evaluator
from test_utils import gen_plan_str, same_plan_str, gt_plan_str, subset_plan_str, partial_subset_str, split_subsequence_str

chooser_same = initialise_chooser([same_plan])
chooser_gt_plan = initialise_chooser([gt_plan])
chooser_subset_plan = initialise_chooser([subset_plan])

class TestEvaluatePlanQualityParsingAndLength(unittest.TestCase):
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
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_same)
        self.assertEqual(len(evaluator.gt_plan), 6)
        self.assertEqual(len(evaluator.gt_actions), 6)

    def test_format_actions_to_plans(self):
        actions = parse_actions(gen_plan)
        plan = format_actions_to_plan(actions)
        self.assertEqual(gen_plan, plan)

    """
    Test the penalise_length_from_chosen and penalise_length_from_optimal methods
    """
    def test_penalise_length_from_chosen_same_plan(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_same)
        length_score = evaluator.penalise_length_from_chosen(gen_plan)
        self.assertEqual(length_score, 0.0)

    def test_penalise_length_from_chosen_longer_curr_plan(self):
        # subset_plan is shorter than gen_plan
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_subset_plan)
        length_score = evaluator.penalise_length_from_chosen(gen_plan)
        length_diff = len(gen_plan) - len(subset_plan)
        self.assertEqual(length_score, length_diff ** 2 / len(subset_plan))
        self.assertFalse(evaluator.generation_required)

    def test_penalise_length_from_chosen_shorter_curr_plan(self):
        # gt_plan is longer than gen_plan
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_gt_plan)
        length_score = evaluator.penalise_length_from_chosen(gen_plan)
        length_diff = len(gen_plan) - len(gt_plan)
        self.assertEqual(length_score, 2 * (length_diff ** 2) / len(gen_plan))
        self.assertTrue(evaluator.generation_required)

    def test_penalise_length_from_opt_same_plan(self):
        # Chosen plan is longer, but optimal plan is the same
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_gt_plan)
        length_score = evaluator.penalise_length_from_optimal(same_plan)
        self.assertEqual(length_score, 0.0)

    def test_penalise_length_from_opt_longer_curr_plan(self):
        # Chosen plan is same, but optimal plan is longer (not actually possible... but chosen should not matter)
        evaluator = EvaluatePlan(gen_plan_str, subset_plan_str, chooser)
        length_score = evaluator.penalise_length_from_optimal(gen_plan)
        self.assertEqual(length_score, (len(gen_plan) - len(subset_plan)) / len(subset_plan))
    
    def test_penalise_length_from_opt_shorter_curr_plan(self):
        # Chosen plan is same, but optimal plan is shorter
        evaluator = EvaluatePlan(gen_plan_str, gt_plan_str, chooser)
        length_score = evaluator.penalise_length_from_optimal(gen_plan)
        self.assertEqual(length_score, -2 * (len(gen_plan) - len(gt_plan)) / len(gt_plan))


    """
    Test functions for action similarity
    """
    def test_check_action_similarity_same_plan(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_same)
        gen_actions = evaluator.curr_actions
        curr_actions, gt_actions, _ = evaluator.check_action_correctness(gen_actions)
        # Should all be 0, as no actions should remain in lists
        self.assertEqual(len(curr_actions), 0)
        self.assertEqual(len(gt_actions), 0)

        similarity_score = evaluator.check_action_similarity(curr_actions, gt_actions)
        self.assertEqual(similarity_score, 0)

    def test_check_action_similarity_pure_subset_plan(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_subset_plan)
        gen_actions = evaluator.curr_actions
        curr_actions, gt_actions, _ = evaluator.check_action_correctness(gen_actions)
        # 2 actions should remain in Curr action and 0 in GT actions
        self.assertEqual(len(curr_actions), 2)
        self.assertEqual(len(gt_actions), 0)

        # Should be 0 because nothing to compare to
        similarity_score = evaluator.check_action_similarity(curr_actions, gt_actions)
        self.assertEqual(similarity_score, 0)

    def test_check_action_similarity_pure_subsequence_plan(self):
        ch = initialise_chooser([subsequence])
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, ch)
        gen_actions = evaluator.curr_actions
        curr_actions, gt_actions, _ = evaluator.check_action_correctness(gen_actions)
        # 1 action should remain in Curr action and 0 in GT actions
        self.assertEqual(len(curr_actions), 1)
        self.assertEqual(len(gt_actions), 0)

        # Should be 0 because nothing to compare to
        similarity_score = evaluator.check_action_similarity(curr_actions, gt_actions)
        self.assertEqual(similarity_score, 0)

    def test_check_action_similarity_different_plan(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser_gt_plan)
        gen_actions = evaluator.curr_actions
        curr_actions, gt_actions, _ = evaluator.check_action_correctness(gen_actions)
        # 2 actions should remain in Curr action and 4 in GT actions
        self.assertEqual(len(curr_actions), 2)
        self.assertEqual(len(gt_actions), 4)

        action_pairs = evaluator.get_action_pairs(curr_actions, gt_actions)
        self.assertEqual(len(action_pairs), 2)  # ['(pick-up a)', '(stack a c)'] and ['(pick-up c)', '(stack c b)']

        # Score should be given for matching pairs + "a" repeating in "stack"
        similarity_score = evaluator.check_action_similarity(curr_actions, gt_actions)
        print(similarity_score)
        self.assertEqual(similarity_score, (0.75 + 1) / len(gt_plan) + len(action_pairs) / (2 * len(gt_actions)))

class TestEvaluatePlanQualityConsistentVariableSwap(unittest.TestCase):
    """
    Tests for scoring of consistent variable swap in variable mapping of plan
    """

    def test_get_variable_lists_1_variable(self):
        actions = [["stack", "a"]]
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(actions)

        self.assertEqual(len(variables), 1)
        self.assertIn("a", variables)

    def test_get_variable_lists_gen_plan(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(evaluator.curr_actions)

        self.assertEqual(len(variables), 3)
        self.assertIn("a", variables)
        self.assertIn("b", variables)
        self.assertIn("c", variables)

    def test_get_variable_lists_many_variable(self):
        actions = [['unstack', 'a', 'b'], ['put-down', 'c'], ['pick-up', 'd'], ['stack','e', 'f'], ['pick-up', 'g)']]
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(actions)

        self.assertEqual(len(variables), 7)

    def test_count_swaps_2_vars(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "a", "c": "c"}
        swaps = evaluator.count_swaps(mapping)
        self.assertEqual(swaps, 2)

    def test_count_swaps_circular_swap(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "c", "c": "a"}
        swaps = evaluator.count_swaps(mapping)
        self.assertEqual(swaps, 3)

    def test_swap_variables_2_vars(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "a", "c": "c"}
        new_plan = evaluator.swap_variables(mapping)
        self.assertTrue(len(gen_plan) == len(new_plan))
        self.assertEqual(new_plan, [['unstack', 'c', 'b'], ['put-down', 'c'], ['pick-up', 'b'], ['stack', 'b', 'c'], ['pick-up', 'a'], ['stack', 'a', 'b']])

    def test_swap_variables_circular_swap(self):
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        mapping = {"a": "b", "b": "c", "c": "a"}
        new_plan = evaluator.swap_variables(mapping)
        self.assertTrue(len(gen_plan) == len(new_plan))
        self.assertEqual(new_plan, [['unstack', 'a', 'b'], ['put-down', 'a'], ['pick-up', 'b'], ['stack', 'b', 'a'], ['pick-up', 'c'], ['stack', 'c', 'b']])

    def test_search_optimal_internal_swap_same_plan(self):
        # Optimal swap should be no swap because they are identical
        evaluator = EvaluatePlan(gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(evaluator.curr_actions)
        optimal_plan, _ = evaluator.search_optimal_internal_swap(variables)

        self.assertEqual(optimal_plan, evaluator.curr_actions)

    def test_search_optimal_internal_swap_gt_plan(self):
        # Optimal swap should be no swap
        evaluator = EvaluatePlan(gen_plan_str, gt_plan_str, chooser_gt_plan)
        variables = get_variable_lists(evaluator.curr_actions)
        optimal_plan, _ = evaluator.search_optimal_internal_swap(variables)

        self.assertEqual(optimal_plan, evaluator.curr_actions)

    def test_search_optimal_internal_swap_swapped_plan(self):
        # Optimal swap should be actions of OG gen_plan
        swapped_gen_plan_str = "(unstack c b)\n(put-down c)\n(pick-up b)\n(stack b c)\n(pick-up a)\n(stack a b)"
        evaluator = EvaluatePlan(swapped_gen_plan_str, same_plan_str, chooser)
        variables = get_variable_lists(evaluator.curr_actions)
        optimal_plan, _ = evaluator.search_optimal_internal_swap(variables)
        # The OG actions of gen_plan before the swap!
        self.assertEqual(optimal_plan, [['unstack', 'c', 'a'], ['put-down', 'c'], ['pick-up', 'a'], ['stack', 'a', 'c'], ['pick-up', 'b'], ['stack', 'b', 'a']])


    def test_consistent_variable_swap(self):
        pass

class TestEvaluatePlanQuality(unittest.TestCase):
    """
    Test full evaluation of the plan
    """
    def test_chooses_same_plan(self):
        """
        Test that the evaluation score matches the expected score when the same plan is chosen
        """
        self.assertEqual(chooser.get_gt_plan(), same_plan)

        # Only one plan to choose from, so it should choose the same plan
        evaluator_same = initialise_evaluator(same_plan_str, chooser_same)

        # All plans to choose from, it should still choose the same plan (identical)
        evaluator = initialise_evaluator(same_plan_str, chooser)

        self.assertEqual(evaluator.get_score(), evaluator_same.get_score())

    def test_full_evaluation_same_plan(self):
        evaluator = initialise_evaluator(same_plan_str, chooser_same)
        self.assertAlmostEqual(evaluator.get_score(), 9.5, 2)

    def test_full_evaluation_different_plan(self):
        evaluator = initialise_evaluator(gt_plan_str, chooser_gt_plan)
        self.assertAlmostEqual(evaluator.get_score(), 5.7, 1)

if __name__ == '__main__':
    unittest.main()
