import unittest

from scripts.plan_recovery.common import get_largest_subsequence, get_largest_subset
from scripts.plan_recovery.choose_gt_plan import ChooseGTPlan
from test_utils import gen_plan, same_plan, subsequence, subset_plan, split_subset, partial_subset, split_subsequence, partial_subsequence, auxiliary, chooser

class TestChooseGTPlan(unittest.TestCase):
    # Tests for get_largest_subsequence and largest_common_subset
    def test_largest_common_same_plan(self):
        subseq = get_largest_subsequence(gen_plan, same_plan)
        subset = get_largest_subset(gen_plan, same_plan)

        self.assertEqual(subseq, same_plan)
        self.assertEqual(len(subseq), len(same_plan))

        self.assertEqual(subset, same_plan)
        self.assertEqual(len(subset), len(same_plan))

    def test_largest_common_pure_subsequence(self):
        subseq = get_largest_subsequence(gen_plan, subsequence)
        subset = get_largest_subset(gen_plan, subsequence)

        self.assertEqual(subseq, subsequence)
        self.assertEqual(len(subseq), len(subsequence))

        self.assertEqual(subset, ['(stack a c)', '(pick-up b)', '(stack b a)'])
        self.assertEqual(len(subset), len(subsequence) - 2)

    def test_largest_common_pure_subset(self):
        subseq = get_largest_subsequence(gen_plan, subset_plan)
        subset = get_largest_subset(gen_plan, subset_plan)

        self.assertEqual(subseq, subset_plan)
        self.assertEqual(len(subseq), len(subset_plan))

        self.assertEqual(subset, subset_plan)
        self.assertEqual(len(subset), len(subset_plan))

    def test_largest_common_partial_subsequence(self):
        subseq = get_largest_subsequence(gen_plan, partial_subsequence)
        subset = get_largest_subset(gen_plan, partial_subsequence)

        self.assertEqual(subseq, ['(unstack c a)', '(put-down c)', '(pick-up b)', '(stack b a)'])
        self.assertEqual(len(subseq), len(partial_subsequence) - 4)

        self.assertEqual(subset, ['(unstack c a)', '(put-down c)'])
        self.assertEqual(len(subset), 2.0)

    def test_largest_common_partial_subset(self):
        gt_plan = ['(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)', '(unstack c a)', '(put-down c)']  # gen_plan, replaced last 2 actions
        subseq = get_largest_subsequence(gen_plan, gt_plan)
        subset = get_largest_subset(gen_plan, gt_plan)

        self.assertEqual(subseq, ['(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)'])
        self.assertEqual(len(subseq), len(gt_plan) - 2)

        self.assertEqual(subset, ['(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)'])
        self.assertEqual(len(subset), len(gt_plan) - 2)

    def test_largest_common_split_subset(self):
        gen_plan = ['(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)']
        subseq = get_largest_subsequence(gen_plan, split_subset)
        subset = get_largest_subset(gen_plan, split_subset)

        self.assertEqual(subseq, ['(unstack c a)', '(put-down c)', '(pick-up b)', '(stack b a)'])
        self.assertEqual(len(subseq), len(split_subset) - 1)

        self.assertEqual(subset, ['(unstack c a)', '(put-down c)'])
        self.assertEqual(len(subset), 2)

    # Tests for penalise_length_diff
    def test_penalise_length_diff_same_length(self):
        gt_plan = ["0", "1", "2", "3", "4", "5"]  # len = 6, gen_plan = 6
        score = chooser.penalise_length_diff(gt_plan, gt_plan)
        self.assertEqual(score, 0)

    def test_penalise_length_diff_curr_shorter(self):
        gt_plan = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]  # len = 12, gen_plan = 6
        score = chooser.penalise_length_diff(gen_plan, gt_plan)
        self.assertEqual(score, -12)

    def test_penalise_length_diff_curr_longer(self):
        gt_plan = ["0", "1", "2", "3"]  # len = 4, gen_plan = 6
        score = chooser.penalise_length_diff(gen_plan, gt_plan)
        self.assertEqual(score, -2)

    def test_choose_gt_plan(self):
        chooser.choose_gt_plan()
        gt_plan = chooser.get_gt_plan()
        self.assertEqual(gt_plan, same_plan)

    def test_choose_gt_plan_subsequence(self):
        # Removed the same plan
        gt_plans = [subsequence, subset_plan, split_subset, partial_subset, split_subsequence,
                    partial_subsequence]
        temp_chooser = ChooseGTPlan(gen_plan, gt_plans)
        temp_chooser.choose_gt_plan()
        gt_plan = temp_chooser.get_gt_plan()

        self.assertEqual(gt_plan, subsequence)

    def test_choose_gt_plan_auxiliary(self):
        # Removed the same plan,
        # TODO should actually choose the subset plan as the GT plan, because the generated plan contains
        #  all the actions in the gt plan, so it doesn't require regeneration
        gt_plans = [subsequence, subset_plan, split_subset, partial_subset, split_subsequence,
                    partial_subsequence, auxiliary]
        # gt_plans = [subset_plan, split_subset, partial_subset, auxiliary]
        temp_chooser = ChooseGTPlan(gen_plan, gt_plans)
        temp_chooser.choose_gt_plan()
        gt_plan = temp_chooser.get_gt_plan()
        print(f"Gen plan: {gen_plan}")
        print(f"GT plan: {gt_plan}")
        print(f"Subset: {subset_plan}")

        self.assertEqual(gt_plan, auxiliary)

    def test_choose_gt_plan_short_curr_plan(self):
        # A subset of the gt plan "subset_plan", but missing 1 action
        # Removed the same plan
        curr_plan = ['(pick-up a)', '(stack a c)', '(pick-up b)']
        gt_plans = [subsequence, subset_plan, split_subset, partial_subset, split_subsequence,
                    partial_subsequence, auxiliary]
        temp_chooser = ChooseGTPlan(curr_plan, gt_plans)
        temp_chooser.choose_gt_plan()
        gt_plan = temp_chooser.get_gt_plan()

        self.assertEqual(gt_plan, subset_plan)

    def test_choose_gt_plan_partial_subset(self):
        # Includes 2 wrong actions at the end but same length as partial_subset
        # Removed the same plan
        curr_plan = ['(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)', '(unstack a b)', '(put-down b)']
        gt_plans = [subsequence, subset_plan, split_subset, partial_subset, split_subsequence,
                    partial_subsequence, auxiliary]
        temp_chooser = ChooseGTPlan(curr_plan, gt_plans)
        temp_chooser.choose_gt_plan()
        gt_plan = temp_chooser.get_gt_plan()

        self.assertEqual(gt_plan, partial_subset)

if __name__ == '__main__':
    unittest.main()