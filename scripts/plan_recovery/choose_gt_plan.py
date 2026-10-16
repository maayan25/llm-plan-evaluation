# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to choost the ground truth plan against which the generated plan should be evaluated.
# This is done by comparing the generated plan to all possible ground truth plans and choosing the one with the smallest
# cost, and the largest subsequence
import json
import os

from common import get_largest_subset, get_largest_subsequence
from generate_gt_plans import generate_gt_plans
from utils import parse_plan, get_llm_plans_from_json, unparse_plan

# Please run the file from its directory or modify the paths
current_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = os.path.abspath(os.path.join(project_dir, os.pardir))

class ChooseGTPlan:
    """
    Favourable:
    1. Same plan
    2. Plan with largest subsequence if includes init and goal states
    3. Plan with largest subset (consecutive actions)
    4. Plan with largest subsequence if much longer than largest subset
    5. Plan with smallest cost
    """
    def __init__(self, generated_plan: str, gt_plans: list[str]):
        self.llm_plan = parse_plan(generated_plan)
        self.gt_plans = [parse_plan(plan) for plan in gt_plans]
        self.gt_plan = []
        self.score = 0

    def get_score(self) -> float:
        """
        Get the score of the generated plan against the chosen ground truth plan
        :return: the score of the generated plan against the chosen ground truth plan
        """
        return self.score

    def get_gt_plan(self) -> str:
        """
        Get the chosen ground truth plan against which the generated plan was evaluated, as a string
        :return: the chosen ground truth plan
        """
        return unparse_plan(self.gt_plan)

    def choose_gt_plan_heuristic(self) -> None:
        """
        Choose the ground truth plan against which to evaluate the generated plan, and set the class's gt_plan and
        score attributes accordingly.
        """
        plan_scores = self.get_plan_scores()
        # best_plan = max(plan_scores, key=plan_scores.get)  # get the plan with the highest score (tuple)
        # Prioritise plans that do not require generation, then choose the one with the highest score
        best_plan = max(
            plan_scores,
            key=lambda gt_candidate: (
                # not self.is_generation_required(self.llm_plan, list(gt_candidate)),
                plan_scores[gt_candidate],
                -self.get_plan_cost(gt_candidate)
            ),
        )

        self.score = plan_scores[best_plan]
        self.gt_plan = list(best_plan)

    def get_plan_scores(self) -> dict:
        """
        Get the scores of the generated plan against all ground truth plans
        :return: a dictionary of the scores of the generated plan against all ground truth plans
        """
        plan_scores = {}

        for gt_plan in self.gt_plans:
            if self.llm_plan == gt_plan:
                self.gt_plan = gt_plan
                self.score = float(2 * len(gt_plan))
                return {tuple(gt_plan): self.score}

            score = self.compute_plan_score(self.llm_plan, gt_plan)
            plan_scores[tuple(gt_plan)] = score

        return plan_scores

    def compute_plan_score(self, curr_plan, gt_plan) -> float:
        """
        Compute the similarity score of the generated plan against the ground truth plan
        :param curr_plan: the generated plan to evaluate
        :param gt_plan: the ground truth plan to compare against
        :return: a score of similarity between the two plans
        """
        if len(curr_plan) == 0:
            return float(-len(gt_plan))

        subsequence_score = len(get_largest_subsequence(curr_plan, gt_plan))
        subset_score = len(get_largest_subset(curr_plan, gt_plan))
        length_penalty = self.compute_length_diff_penalty_size(curr_plan, gt_plan) / len(curr_plan)

        # Simple heuristic to determine if the subsequence is much longer than the subset
        # score = len(subseq) + len(subset)  # Equivalent to len(subseq) - len(subset) + len(subset) * 2, which would be the part of the subseq that is not in the subset, plus the subset * 2 (because it is more important than the subsequence)
        lcs_score = (1/3 * subsequence_score) + (2/3 * subset_score)
        score = lcs_score - length_penalty  # penalise for length difference

        return score

    def is_generation_required(self, llm_plan, gt_plan) -> bool:
        """
        Determine if a new plan needs to be generated based on the cost of the current plan and the ground truth plan.
        If the cost of the llm plan is smaller than the cost of the gt plan, then a new plan must be generated.
        :param llm_plan: the generated plan to evaluate
        :param gt_plan: the ground truth plan to compare against
        :return: whether a new plan definitely needs to be generated
        """
        llm_plan_cost = self.get_plan_cost(llm_plan)
        gt_plan_cost = self.get_plan_cost(gt_plan)

        if llm_plan_cost < gt_plan_cost:
            return True
        else:
            return False


    def compute_length_diff_penalty_size(self, curr_plan, gt_plan) -> float:
        """
        Reduce the score of plans depending on their difference from the ground truth plan.
        Penalises shorter plans more - longer plans can be suboptimal, shorter plans are necessarily missing vital actions.
        """
        length_diff = len(curr_plan) - len(gt_plan)

        if length_diff > 0:
            score = length_diff # current plan may include the GT plan
        elif length_diff == 0:
            score = 0 # no penalty if plans are the same length
        else:
            score = -2 * length_diff # current plan may be subset of GT plan, a new plan will definitely need to be generated

        # print(f"Length diff: {length_diff}, score: {score}")
        return abs(score)

    def get_plan_cost(self, plan) -> float:
        """
        Get the cost of the plan. Currently implemented as length of the plan, but can be modified to use a more
        complex cost function if needed.
        :param plan: the plan to evaluate
        :return: the cost of the plan
        """
        return float(len(plan))

def get_gt_plans(domain, instance_id, validate=True) -> list[str]:
    """
    Get the ground truth plans for the given domain and instance id
    :param domain: the domain to get the ground truth plans for
    :param instance_id: the instance id to get the ground truth plans for
    :param validate: whether to get the validated ground truth plans or the original ground truth plans
    :return: a list of ground truth plans
    """
    filename = "valid_plans.json" if validate else "gt_plans.json"
    gt_file_path = os.path.join(os.pardir, os.pardir, "results/gt_plans", domain, filename)
    if not os.path.exists(gt_file_path):
        # Raise an error if the file does not exist, as the ground truth plans should have been generated already
        raise FileNotFoundError(f"Ground truth plans file not found: {gt_file_path}. Please generate the ground truth plans for domain {domain} before running this script.")
        # If the file does not exist, generate the ground truth plans for this domain
        # print(f"Ground truth plans file not found: {gt_file_path}. Generating ground truth plans for domain {domain}, this may take a while...")
        # generated_plans, valid_plans = generate_gt_plans(domain, validate=validate, generate=True)
        # gt_plans = valid_plans if validate else generated_plans
    else:
        with open(gt_file_path, "r") as f:
            data = json.load(f)
            gt_plans = data["instances"][str(instance_id)]
    return gt_plans

def get_gt_plans_for_domain(domain, validate=True) -> dict:
    """
    Get the ground truth plans for the given domain
    :param domain: the domain to get the ground truth plans for
    :return: a dictionary of instance id to ground truth plans
    """
    filename = "valid_plans.json" if validate else "gt_plans.json"
    gt_file_path = os.path.join(os.pardir, os.pardir, "results/gt_plans", domain, filename)
    with open(gt_file_path, "r") as f:
        data = json.load(f)
        gt_plans = data["instances"]
    gt_plans = {int(instance_id): [plan for plan in plans] for instance_id, plans in gt_plans.items()}
    return gt_plans

def save_chosen_gt_plan(domain, instance_id, gt_plan, filename) -> None:
    """
    Save the chosen ground truth plan to a JSON file for the given domain and instance id
    :param domain: the domain to save the ground truth plan for
    :param instance_id: the instance id to save the ground truth plan for
    :param gt_plan: the ground truth plan to save
    """
    gt_file_path = os.path.join(os.pardir, os.pardir, "results/gt_plans", domain, filename)
    with open(gt_file_path, "w") as f:
        data = json.load(f)
        data["chosen_plans"][str(instance_id)] = unparse_plan(gt_plan)
        json.dump(data, f)
    print(f"Saved chosen GT plan for instance {instance_id} to {gt_file_path}")

def save_chosen_gt_plans_for_domain(domain, chosen_plans, output_file) -> None:
    """
    Save the chosen ground truth plans to a JSON file for the given domain
    :param domain: the domain to save the ground truth plans for
    :param chosen_plans: a dictionary of instance id to ground truth plans
    """
    gt_file_path = os.path.join(os.pardir, os.pardir, "results/gt_plans", domain, output_file)
    os.makedirs(os.path.dirname(gt_file_path), exist_ok=True)

    # Get the config = the three parent firectories of the output file path, e.g. "blocksworld_3/qcode/_zero_shot"
    config = os.path.join(*os.path.dirname(gt_file_path).split(os.sep)[-3:])

    if os.path.exists(gt_file_path):
        with open(gt_file_path, "r") as f:
            data = json.load(f)
    else:
        data = {"config": config, "chosen_plans": {}}

    for instance_id, gt_plan in chosen_plans.items():
        data["chosen_plans"][str(instance_id)] = unparse_plan(gt_plan)

    with open(gt_file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_chosen_gt_plan(domain, instance_id, validate=True) -> list[str]:
    """
    Get the chosen ground truth plan for the given domain and instance id
    :param domain: the domain to get the ground truth plan for
    :param instance_id: the instance id to get the ground truth plan for
    :return: the chosen ground truth plan
    """
    filename = "valid_plans.json" if validate else "gt_plans.json"
    gt_file_path = os.path.join(os.pardir, os.pardir, "results/gt_plans", domain, filename)
    with open(gt_file_path, "r") as f:
        data = json.load(f)
        gt_plan = data["chosen_plans"][str(instance_id)]
    gt_plan = parse_plan(gt_plan)
    return gt_plan

def choose_gt_plans_for_domain(domain, generated_plans, validate=True) -> dict[str, str]:
    """
    Choose the ground truth plans for the given domain and save them to a JSON file
    :param domain: the domain to choose the ground truth plans for
    :param generated_plans: a dictionary of instance id to generated plans
    :return: a dictionary of instance id to chosen ground truth plans
    """
    gt_plans = get_gt_plans_for_domain(domain, validate=validate)
    chosen_plans = {}
    for instance_id, generated_plan in generated_plans.items():
        # generated_plan = parse_plan(generated_plan)
        chooser = ChooseGTPlan(generated_plan, gt_plans[instance_id])
        chooser.choose_gt_plan_heuristic()
        chosen_plans[instance_id] = chooser.get_gt_plan()
        print(f"Chosen plan for instance {instance_id}: {chooser.get_gt_plan()} with score: {chooser.get_score()}")
        print(f"LLM generated plan: {generated_plan}")
        # save_chosen_gt_plan(domain, instance_id, chosen_plans[instance_id], validate=validate)
    return chosen_plans

def test_pipeline():
    domain = "blocksworld_3"
    instance_id = 84
    gt_plans = get_gt_plans(domain, instance_id)
    # chooser = ChooseGTPlan(["a", "b", "c", "d"], [["a", "b", "c"], ["a", "b", "c", "e"], ["a", "b", "c", "d", "e", "f"]])
    # chooser = ChooseGTPlan(["(pick-up a)", "(stack a b)", "(pick-up b)", "(stack b a)"], gt_plans)
    chooser = ChooseGTPlan("(pick-up c) (put-down a) (unstack a b) (stack a c)", gt_plans)

    chooser.choose_gt_plan_heuristic()
    print(f"Chosen plan: {chooser.get_gt_plan()} with score: {chooser.get_score()}")
    save_chosen_gt_plan(domain=domain, instance_id=instance_id, gt_plan=chooser.get_gt_plan(), filename="chosen_gt_plans.json")

def main():
    temp = 0.1
    top_p = 1

    domain = "blocksworld_3"
    model = "qcode"
    task = "_zero_shot"  # "_one_shot", "_state_tracking", "_pddl", "_zero_shot"

    filepath = f"{parent_dir}/LLM-Planning-PlanBench/llm_planning_analysis/results/{domain}/temp_{temp}_top_p_{top_p}/{model}/task_1_plan_generation{task}.json"
    validate = True  # False to choose GT plans from non-validated plans
    generated_plans = get_llm_plans_from_json(filepath)

    chosen_plans = choose_gt_plans_for_domain(domain, generated_plans, validate=validate)
    output_file = "validated_chosen_gt_plans.json" if validate else "chosen_gt_plans.json"
    output_file = f"{model}/{task}/" + output_file
    save_chosen_gt_plans_for_domain(domain, chosen_plans, output_file)

if __name__ == "__main__":
    main()