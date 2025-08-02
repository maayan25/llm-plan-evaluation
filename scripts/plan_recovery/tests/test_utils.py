from scripts.plan_recovery.choose_gt_plan import ChooseGTPlan
from scripts.evaluate_plan_quality import EvaluatePlan
from scripts.plan_recovery.evaluate_plan_per_action import EvaluatePlanPerAction

gen_plan = ['(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)']
same_plan = [ '(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)']
gt_plan = ['(unstack c a)', '(put-down c)', '(unstack a b)', '(put-down a)', '(pick-up b)', '(stack b a)', '(pick-up c)', '(stack c b)']
subsequence = ['(unstack c a)', '(put-down c)', '(stack a c)', '(pick-up b)', '(stack b a)']  # non consecutive subsequence
subset_plan = ['(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)']  # a pure subset of gen_plan
split_subset = ['(unstack c a)', '(put-down c)', '(pick-up c)', '(pick-up b)', '(stack b a)']  # 2 subsets of gen_plan + 1 wrong action in the middle
partial_subset = ['(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)', '(unstack c b)', '(put-down a)']  # a subset of gen_plan + 2 wrong actions at the end
split_subsequence = ['(unstack c a)', '(put-down c)', '(pick-up b)', '(stack b a)', '(pick-up b)', '(stack b a)']  # subseq of gen_plan + 2 (wrong position) actions in the middle
partial_subsequence = ['(unstack c a)', '(put-down c)', '(pick-up c)', '(stack c a)', '(pick-up b)', '(stack b a)', '(pick-up b)', '(stack b a)']  # subseq of gen_plan + 2 actions in the middle + 2 actions at the end
auxiliary = ['(unstack c a)', '(put-down c)', '(pick-up a)', '(stack a c)', '(pick-up b)', '(stack b a)', '(pick-up c)', '(stack c b)']  # gen_plan + 2 actions at the end

gen_plan_str = "(unstack c a)\n(put-down c)\n(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)"
same_plan_str = "(unstack c a)\n(put-down c)\n(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)"
gt_plan_str = "(unstack c a)\n(put-down c)\n(unstack a b)\n(put-down a)\n(pick-up b)\n(stack b a)\n(pick-up c)\n(stack c b)"
subsequence_str = "(unstack c a)\n(put-down c)\n(stack a c)\n(pick-up b)\n(stack b a)"
subset_plan_str = "(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)"
split_subset_str = "(unstack c a)\n(put-down c)\n(pick-up c)\n(pick-up b)\n(stack b a)"
partial_subset_str = "(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)\n(unstack c b)\n(put-down a)"
split_subsequence_str = "(unstack c a)\n(put-down c)\n(pick-up b)\n(stack b a)\n(pick-up b)\n(stack b a)"
partial_subsequence_str = "(unstack c a)\n(put-down c)\n(pick-up c)\n(stack c a)\n(pick-up b)\n(stack b a)\n(pick-up b)\n(stack b a)"
auxiliary_str = "(unstack c a)\n(put-down c)\n(pick-up a)\n(stack a c)\n(pick-up b)\n(stack b a)\n(pick-up c)\n(stack c b)"

gt_plans = [same_plan, subsequence, subset_plan, split_subset, partial_subset, split_subsequence, partial_subsequence]

chooser = ChooseGTPlan(gen_plan, gt_plans)
chooser.choose_gt_plan()

def initialise_chooser(plans: list[list[str]]) -> ChooseGTPlan:
    ch = ChooseGTPlan(gen_plan, plans)
    ch.choose_gt_plan()
    return ch

def initialise_evaluator(gt: str, ch: ChooseGTPlan) -> EvaluatePlan:
    evaluator = EvaluatePlan(gen_plan_str, gt, ch)
    evaluator.evaluate_plan()
    return evaluator

def initialise_evaluator_per_action(curr: str, gt: str, ch: ChooseGTPlan) -> EvaluatePlanPerAction:
    evaluator = EvaluatePlanPerAction(curr, gt, ch)
    evaluator.run_evaluation()
    return evaluator