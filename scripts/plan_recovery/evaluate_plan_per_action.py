# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to evaluate aspects of the invalid plan and provide an overall weighted score
# for finer granularity of evaluation, rather than binary True or False

import os
import itertools
from copy import deepcopy
import uuid

from evaluate_plan import Plan
from common import get_variable_lists, make_list_hashable
from utils import (parse_plan, parse_actions, format_actions_to_plan, get_plans_from_json,
                                         set_configuration)
from choose_gt_plan import ChooseGTPlan

# Please run the file from its directory or modify the paths
current_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = os.path.abspath(os.path.join(project_dir, os.pardir))


class EvaluatePlanPerAction:
    """
    Class to evaluate a plan based on a ground truth plan.
    """
    def __init__(self, curr_plan: str, optimal_gt_plan: str, validity=False, domain="blocksworld_3", instance_id=2, unique_id=None):
        self.is_valid = validity
        # self.chooser: ChooseGTPlan = chooser
        # self.gt_plan: list[str] = chooser.get_gt_plan()
        self.gt_plan: list[str] = parse_plan(optimal_gt_plan)
        self.gt_actions: list[list[str]] = parse_actions(self.gt_plan)  # for chosen ground truth plan
        self.curr_plan = None
        self.domain_name = domain
        self.instance_id = instance_id
        self.unique_id = unique_id

        if self.is_valid:
            print("Valid plan was provided.")

        if curr_plan is not None:
            print(f"Evaluating plan for instance {instance_id} with unique ID {unique_id}")
            self.curr_plan = Plan(parse_plan(curr_plan), self.gt_plan, self.is_valid, self.domain_name, self.instance_id, unique_id=self.unique_id)

            # print(f"Current plan: {self.curr_plan.get_plan()} and GT chosen plan: {self.gt_plan}")
        else:
            print("No plan was generated for this instance.")
            self.final_score = 0.0
            return

        if optimal_gt_plan is not None:
            self.optimal_gt_plan: list[str] = parse_plan(optimal_gt_plan)
        else:
            # TODO change this as might give higher score if uses chosen GT plan
            print("No optimal GT plan was provided, using the chosen GT plan instead.")
            self.optimal_gt_plan = self.gt_plan

        self.generation_required = False       # whether another plan generation is definitely required
        self.consistent_variable_swap = False  # whether a variable swap improves the generated plan
        self.offset_improval = False           # whether an offset improves the generated plan
        self.optimality = {}                   # data on the optimal swap and offset

        self.possible_plans: list[Plan] = []  # list of possible plans to evaluate
        self.candidate_plan: Plan = self.curr_plan    # the plan that has higher chances of being fixed by the LLM's new generation
        self.valid_plan: list[str] = []                          # a variation of the plan that is valid according to the ground truth

        # self.score = 5.0  # score for curr_plan
        self.potential_score = 5.0  # score for candidate plan / the highest score without regeneration
        self.final_score = 5.0  # score that takes both into account

    def get_final_score(self) -> float:
        """
        Get the final score of the plan evaluation
        :return: the final score of the plan evaluation
        """
        return self.final_score

    def penalise_length_from_chosen(self, plan: list[str]) -> float:
        """
        Reduce the score of plans depending on their difference from the ground truth plan.
        Penalises shorter plans more - longer plans can be suboptimal, shorter plans are necessarily missing vital actions.
        The score is normalised by the length of the ground truth plan (the generated plan can be empty).
        :param plan: the plan to evaluate, should be of type list[str], but can also be actions (list[list[str]])
        """
        length_diff = len(plan) - len(self.gt_plan)

        if length_diff > 0:
            score = length_diff ** 2 / len(self.gt_plan) # current plan may include the GT plan
        elif length_diff == 0:
            score = 0 # no penalty if plans are the same length
        else:
            score = 2 * (length_diff ** 2)  / len(self.gt_plan) # WAS / len(plan) # current plan may be subset of GT plan, a new plan will definitely need to be generated
            self.generation_required = True
        return score

    # def penalise_length_from_optimal(self, plan: list[str]) -> float:
    #     """
    #     Penalise the length of the plan from the optimal GT plan, i.e. penalise cost with actions weight 1. Score
    #     is positive, as it is a penalty, and it is normalised by the length of the optimal GT plan.
    #     :param plan: the curr plan to evaluate
    #     :return: the score to penalise the plan (positive)
    #     """
    #     # TODO implement with actual costs if using them
    #     # Returns a positive score for consistency with penalise_length_from_chosen
    #     score = -1 * self.chooser.penalise_length_diff(plan, self.optimal_gt_plan) / len(self.optimal_gt_plan)
    #     return score

    def check_consistent_variable_swap(self) -> None:
        """
        Function to check if in actions that are not correct and in the right position, if a variable is swapped with
        another consistently across functions, more actions are correct and in the right position.
        If there is a successful full or partial swap, increase score accordingly, and save plan as candidate plan.
        :return:
        """
        can_be_valid = True

        # Populate lists from self.curr_actions and self.gt_actions (want to check actions that are in wrong position as well)
        curr_variables = get_variable_lists(self.curr_plan.get_actions())
        gt_variables = get_variable_lists(self.gt_actions)

        for var in gt_variables:
            if var not in curr_variables:
                # if a variable is in gt but not in curr, swapping inside curr will not create a valid plan
                can_be_valid = False

        # Evaluate each possible plan and score accordingly
        self.optimality = self.search_optimal_internal_swap(curr_variables)
        optimal_plan = self.optimality["optimal_plan"]
        best_score = self.optimality["best_score"]

        if not can_be_valid:
            # TODO try swapping with variables in gt_actions?? give lower score than if swapped within plan
            pass

        # If there is a consistent solution higher than a threshold, set it as the candidate plan and set boolean to true
        self.candidate_plan = optimal_plan
        self.potential_score = best_score

    def search_optimal_internal_swap(self, variables) -> dict:
        """
        Search for optimal internal swap using beam search to reduce complexity.
        :param variables: list of variables in the current plan
        :param beam_size: how many top candidates to keep per beam iteration
        :return: best plan and its score
        """
        optimal_plan: Plan = self.curr_plan
        best_score = optimal_plan.get_score()
        valid_plan = None
        valid_swap = {}
        optimal_swap = {}
        optimal_offset = 0
        evaluated_plans = {}

        if len(variables) < 7:
            candidate_mappings = [
                dict(zip(variables, perm)) for perm in itertools.permutations(variables)
            ]
            # print(f"Type of candidate mappings: {type(candidate_mappings)} for small mappings and {len(candidate_mappings)} mappings")
            # print(f"Where each mapping is of type: {type(candidate_mappings[0])} and {len(candidate_mappings[0])} variables")
        else:
            # Build all mappings that make at least one action match the GT
            candidate_mappings = self.get_candidate_mappings_from_gt_matches(variables)
            # print(f"Type of candidate mappings: {type(candidate_mappings)} for large mappings")

        if not candidate_mappings:
            # Fallback to identity mapping if nothing found
            candidate_mappings = [{v: v for v in variables}]

        for mapping in candidate_mappings:
            actions = self.swap_variables(mapping)
            plan_list = format_actions_to_plan(actions)
            plan_key = tuple(plan_list)

            if plan_key in evaluated_plans:
                plan, score = evaluated_plans[plan_key]
            else:
                plan = Plan(plan_list, self.gt_plan, self.is_valid, self.domain_name, self.instance_id,
                            to_simulate=False, unique_id=f"{self.unique_id}_{uuid.uuid4()}" if self.unique_id else None)

                offset = self.search_optimal_offset(plan)
                if offset != 0:
                    new_actions = self.shift_actions(plan.get_actions(), offset)
                    plan_list = format_actions_to_plan(new_actions)
                    plan = Plan(plan_list, self.gt_plan, self.is_valid, self.domain_name, self.instance_id,
                                to_simulate=False, unique_id=f"{self.unique_id}_{uuid.uuid4()}" if self.unique_id else None)
                    optimal_offset = offset

                if not plan.is_valid:
                    plan.check_plan_validity()
                if plan.is_valid:
                    if valid_plan is None or len(valid_plan.get_plan()) > len(plan.get_plan()):
                        valid_plan = plan
                        valid_swap = mapping
                    self.evaluate_valid_plan(plan)
                else:
                    self.evaluate_plan(plan)

                penalty = self.count_swaps(mapping) / len(variables)
                plan.decrease_score(penalty)
                score = plan.get_score()
                self.possible_plans.append(plan)
                evaluated_plans[plan_key] = (plan, score)

            if score > best_score:
                best_score = score
                optimal_swap = mapping
                optimal_plan = deepcopy(plan)
                if optimal_plan.get_actions() != self.curr_plan.get_actions():
                    self.consistent_variable_swap = True

        if valid_plan is not None and valid_plan.get_score() > best_score:
            optimal_plan = valid_plan
            optimal_swap = valid_swap

        # optimal_plan.states = optimal_plan.simulate_states()
        # Avoid re-evaluating the same plan
        if optimal_plan.plan != self.curr_plan.plan:
            states = optimal_plan.simulate_states()
            optimal_plan.states = tuple(deepcopy(states))

        return {
            "optimal_plan": optimal_plan,
            "best_score": best_score,
            "optimal_mapping": optimal_swap,
            "optimal_offset": optimal_offset
        }

    def count_swaps(self, mapping: dict[str: str]) -> float:
        """
        Count number of swaps in a mapping to be able to penalise swaps
        :return: number of swaps
        """
        swaps = 0
        for key in mapping:
            if key != mapping[key]:
                swaps += 1

        return swaps

    def swap_variables(self, mapping: dict[str: str]) -> list[list[str]]:
        """
        Swap two variables consistently in the current plan
        :param mapping: each variable to its new value
        :return: the new plan with the swapped variables
        """
        new_actions: list[list[str]] = []

        for action in self.curr_plan.get_actions():
            new_action = []
            for item in action:
                if item in mapping: # skip action name and only map variables
                    item = mapping[item]
                new_action.append(item)
            new_actions.append(new_action)

        # print(f"OG actions {self.curr_plan.get_actions()}")
        # print(f"new actions {new_actions}")

        return new_actions

    def search_optimal_offset(self, plan: Plan) -> int:
        """
        Search for the optimal offset in the current plan by maximising the score of the plan, with beam search.
        :param plan: the plan to evaluate
        :return: the optimal offset
        """
        evaluated_plans = {}
        valid_plan = []
        valid_offset = 0
        optimal_offset = 0
        best_score = plan.get_score()

        num_offsets = len(self.curr_plan.get_actions())

        # Action diversity for all shifts
        candidate_offsets = sorted(
            range(num_offsets),
            key=lambda o: -self.offset_heuristic(self.shift_actions(plan.get_actions(), o))
        )

        # Dynamic beam size based on plan length
        beam_size = max(7, min(15, num_offsets)) # 7 for small plans, 15 for huge ones

        top_offsets = candidate_offsets[:beam_size]

        for offset in top_offsets:
            actions = self.shift_actions(plan.get_actions(), offset)
            plan_list = format_actions_to_plan(actions)
            plan_key = make_list_hashable(plan_list)

            if plan_key in evaluated_plans:
                score, is_valid = evaluated_plans[plan_key]
            else:
                shifted_plan = Plan(plan_list, self.gt_plan, self.is_valid, self.domain_name, self.instance_id,
                                    to_simulate=False, unique_id=f"{self.unique_id}_{uuid.uuid4()}" if self.unique_id else None)
                if not shifted_plan.is_valid:
                    shifted_plan.check_plan_validity()

                if shifted_plan.is_valid:
                    if not valid_plan or len(valid_plan) > len(shifted_plan.get_plan()): # a shorter valid plan is better
                        valid_plan = shifted_plan.get_plan()
                        valid_offset = offset
                    self.evaluate_valid_plan(shifted_plan)
                else:
                    self.evaluate_plan(shifted_plan)

                # Penalise number of shifts
                shifted_plan.decrease_score(offset / 2)
                self.possible_plans.append(shifted_plan)

                score = shifted_plan.get_score()
                is_valid = shifted_plan.is_valid
                evaluated_plans[plan_key] = (score, is_valid)

                if score > best_score:
                    best_score = score
                    optimal_offset = offset
                    if optimal_offset != 0:
                        self.offset_improval = True

        if len(valid_plan) != 0:
            # print(f"VALID PLAN FOUND: {valid_plan} with offset {valid_offset}")
            optimal_offset = valid_offset

        return optimal_offset

    def shift_actions(self, actions:list[list[str]], offset: int) -> list[list[str]]:
        """
        Shift the actions in the current plan by a given offset; done in a circular manner, so actions that are shifted
        off the end of the plan are added to the beginning (to maintain the same length)
        :param actions: the actions to shift
        :param offset: how much to shift the actions
        :return: the new plan with the shifted actions
        """
        return actions[offset:] + actions[:offset]

    # def offset_heuristic(self, actions: list[list[str]]) -> float:
    #     tuple_actions = [tuple(action) for action in actions]
    #     return len(set(tuple_actions)) / len(tuple_actions)

    # def offset_heuristic(self, actions: list[list[str]]) -> float:
    #     return random()

    def offset_heuristic(self, shifted_actions: list[list[str]]) -> int:
        """
        Score how well the shifted actions align with the ground truth by comparing action names at each position.
        """
        score = 0
        for i, action in enumerate(shifted_actions):
            if i >= len(self.gt_actions):
                break
            if action[0] == self.gt_actions[i][0]:  # Compare only action names
                score += 1
        return score

    def get_candidate_mappings_from_gt_matches(self, curr_variables) -> list[dict[str, str]]:
        """
        Builds a list of variable mappings that would make at least one action in the generated plan
        match an action in the GT plan.
        """
        mappings = set()

        for gt_action in self.gt_actions:
            gt_name = gt_action[0]
            gt_args = gt_action[1:]

            for gen_action in self.curr_plan.get_actions():
                if gen_action[0] != gt_name:
                    continue
                gen_args = gen_action[1:]
                if len(gt_args) != len(gen_args):
                    continue

                mapping = dict(zip(gen_args, gt_args))
                # Fill identity mapping for other variables in the plan
                for v in curr_variables:
                    if v not in mapping:
                        mapping[v] = v

                # Convert to frozenset to avoid duplicates in set
                mappings.add(frozenset(mapping.items()))

        # Convert back to dicts before returning
        return [dict(m) for m in mappings]

    def compute_final_score(self) -> None:
        # TODO maybe change weights
        if self.curr_plan.get_score() < 0.0:
            self.final_score = -1.0
        else:
            self.final_score = (self.curr_plan.get_score() + self.potential_score) / 2 # '(2 * len(self.curr_plan.get_plan()))

    def evaluate_plan(self, plan: Plan) -> None:
        """
        Compute score between current plan and ground truth plan based on different metrics. If an empty plan or no
        plan was generated, score is set to 0.
        :param plan: the plan to evaluate
        """
        if plan.is_valid:
            self.valid_plan = plan.get_plan()
        # Scoring by length and cost
        chosen_length_score = self.penalise_length_from_chosen(plan.get_plan())
        # opt_length_score = self.penalise_length_from_optimal(plan.get_plan())

        # Scoring by action correctness and similarity
        action_similarity = plan.get_similarity_score()
        # print(f"Action similarity: {action_similarity}, and length penalty: {chosen_length_score}")

        plan.increase_score(action_similarity - chosen_length_score) # WAS - opt_length_score)

    def evaluate_valid_plan(self, plan) -> None:
        """
        Compute score between current plan and ground truth plan based on optimality metrics.
        """
        self.valid_plan = plan.get_plan()

        # Scoring by length and cost
        if len(plan.get_plan()) > len(self.optimal_gt_plan):
            opt_length_score = self.penalise_length_from_chosen(plan.get_plan()) # / len(plan.get_plan())
        else:
            opt_length_score = 0.0

        # print(f"VALID PLAN PENALTY: {opt_length_score} for {len(plan.get_plan())} and {len(self.optimal_gt_plan)}")

        plan.decrease_score(opt_length_score)


    def run_evaluation(self):
        if self.curr_plan is not None and len(self.curr_plan.get_plan()) > 0 and len(self.gt_plan) > 0:
            print(f"Evaluating plan {self.curr_plan.get_plan()} against GT plan {self.gt_plan}")
            if self.is_valid:
                initial_score = 3 * len(self.gt_plan)
                print(f"Initial score for valid plan: {initial_score}")
                self.curr_plan.increase_score(initial_score)  # make initial score 15.0 / 3 * length
                self.evaluate_valid_plan(self.curr_plan)  # penalise on cost
                self.potential_score = 4 * len(self.gt_plan) / len(self.curr_plan.get_plan())  # penalise on length
                # print(f"Potential score for valid plan: {self.potential_score}")
            else:
                self.evaluate_plan(self.curr_plan)

                # Checks of potential improvement for best plan candidate
                self.check_consistent_variable_swap()
                if len(self.valid_plan) != 0:
                    # There is a valid plan that can be used (half of what it would be for an initially valid plan)
                    self.potential_score = 2 * len(self.valid_plan) / len(self.curr_plan.get_plan())
            self.compute_final_score()
        else:
            self.final_score = 0.0

def main():
    # configuration
    project = "llm_planning_analysis"  # llm_planning_analysis / plan-bench
    domain = "blocksworld_3"  # blocksworld_3, logistics
    model = "qcode"  # qwen, llama, qcode, llama_instruct
    task = ""  # "" -> "_one_shot", "_state_tracking", "_pddl", "_zero_shot"
    temp = 0.1
    top_p = 1
    params = set_configuration(model=model, task=task, domain=domain, project=project, temp=temp, top_p=top_p)

    filepath = f"{parent_dir}/LLM-Planning-PlanBench/{project}/results/{domain}/temp_{temp}_top_p_{top_p}/{model}/task_1_plan_generation{task}.json"

    # evaluate all instances
    scores = {}
    validity = {}
    plans = get_plans_from_json(filepath)

    for instance in plans[:10]:
        # Choose the ground truth plan against which to evaluate the generated plan
        chooser = ChooseGTPlan(parse_plan(instance[1]), [parse_plan(instance[2])])  # TODO change to all possible GT plans
        chooser.choose_gt_plan()

        # Run the evaluation
        evaluator = EvaluatePlanPerAction(instance[1], instance[2], instance[3], domain, instance[0]) # removed chooser
        evaluator.run_evaluation()

        if "zero" in task:
           inst_num = instance[0] + 1
        else:
            inst_num = instance[0]
        scores[inst_num] = evaluator.get_final_score()
        validity[inst_num] = instance[3]

    # write results to csv (from root dir of the project)
    results_dir = f"{project_dir}/results/plan_recovery/{project}/{domain}/evaluation_per_action/{model}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = f"{results_dir}/full_evaluation_testing_score_is_len_or_3_len.csv"  # plan_generation.csv
    # write_evaluation_to_csv(output_file, scores, params, validity) # TODO if want to save need to change to append

if __name__ == "__main__":
    main()