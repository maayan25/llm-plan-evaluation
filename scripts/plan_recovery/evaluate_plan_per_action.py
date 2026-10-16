# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to evaluate aspects of the invalid plan and provide an overall weighted score
# for finer granularity of evaluation, rather than binary True or False
import math
import os
from copy import deepcopy
import uuid
from collections import defaultdict

from evaluate_plan import Plan
from common import get_variable_lists, make_list_hashable
from utils import (parse_plan, parse_actions, format_actions_to_plan, get_plans_from_json,
                   set_configuration, get_object_types)
from choose_gt_plan import ChooseGTPlan

import numpy as np
from scipy.optimize import linear_sum_assignment

# Please run the file from its directory or modify the paths
current_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = os.path.abspath(os.path.join(project_dir, os.pardir))


class EvaluatePlanPerAction:
    """
    Class to evaluate a plan based on a ground truth plan.
    """
    def __init__(self, curr_plan: str, optimal_gt_plan: str, gt_plan: str, validity=False, domain="blocksworld_3", instance_id=2, unique_id=None, search_params=None):
        self.is_valid = validity
        self.gt_plan: list[str] = parse_plan(gt_plan)
        self.gt_actions: list[list[str]] = parse_actions(self.gt_plan)  # for chosen ground truth plan
        self.curr_plan = None
        self.domain_name = domain
        self.instance_id = instance_id
        self.unique_id = unique_id
        self.search_params = search_params or {"top_k": 3, "refine_iters": 2, "min_k": 7, "max_k": 25, "tol": 0.1}
        print(f"DEBUG: Search parameters for instance {instance_id} are {self.search_params}")

        if self.is_valid:
            print("Valid plan was provided.")

        if curr_plan is not None:
            print(f"Evaluating plan for instance {instance_id} with unique ID {unique_id}")
            self.curr_plan = Plan(parse_plan(curr_plan), self.gt_plan, self.is_valid, self.domain_name, self.instance_id, unique_id=self.unique_id)
            print(f"Evaluated plan for instance {instance_id}; has score {self.curr_plan.get_score()} and is_valid: {self.curr_plan.is_valid}")
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
        self.var_types = get_object_types(self.curr_plan.problem_file, self.curr_plan.domain_file)
        print(f"DEBUG TYPES: Variable types for domain {self.domain_name} are {self.var_types}")

        # self.score = 5.0  # score for curr_plan
        self.potential_score = 0.0  # score for candidate plan / the highest score without regeneration
        self.final_score = 0.0  # score that takes both into account


    def get_final_score(self) -> float:
        """
        Get the final score of the plan evaluation
        :return: the final score of the plan evaluation
        """
        return round(self.final_score, 4)

    def penalise_length_from_chosen(self, plan: list[str]) -> float:
        """
        Reduce the score of plans depending on their difference from the ground truth plan.
        Penalises shorter plans more - longer plans can be suboptimal, shorter plans are necessarily missing vital actions.
        The penalty is used later as a weight for the plan's score.
        :param plan: the plan to evaluate, should be of type list[str], but can also be actions (list[list[str]])
        """
        if len(plan) == 0 or len(self.gt_plan) == 0:
            return 1.0

        if len(plan) < len(self.optimal_gt_plan):
            penalty = abs(len(plan) - len(self.gt_plan)) / max(len(plan), len(self.gt_plan))
        else:
            penalty = abs(len(plan) - len(self.gt_plan)) / (2 * max(len(plan), len(self.gt_plan)))
            # penalty = len(plan) / (2 * len(self.gt_plan))
        return penalty

    def penalise_length_from_optimal(self, plan: list[str]) -> float:
        """
        Penalise the length of the plan from the optimal GT plan, i.e. penalise cost with actions weight 1. Score
        is positive, as it is a penalty, and it is normalised by the length of the plan.
        :param plan: the curr plan to evaluate
        :return: the score to penalise the plan (positive)
        """
        return len(self.optimal_gt_plan) / len(plan) if len(plan) > len(self.optimal_gt_plan) else 1

    def check_consistent_variable_swap(self) -> tuple[Plan, float]:
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
        return  optimal_plan, best_score

    def get_top_offsets(self, actions):
        """Adaptive beam: everything within `tol` of the best score (at least min_k, at most max_k), skipping offsets that yield identical shifted sequences."""
        scores = self.score_all_shifts(actions)
        order = np.argsort(-scores, kind="stable")
        tol = self.search_params.get("tol", 0.1)
        cutoff = scores[order[0]] - max(1, tol * len(self.gt_actions))
        picked, seen = [], set()
        max_k = self.search_params.get("max_k", 25)
        min_k = self.search_params.get("min_k", 7)
        for o in order:
            if len(picked) >= max_k:
                break
            if len(picked) >= min_k and scores[o] < cutoff:
                break
            key = tuple(map(tuple, self.shift_actions(actions, int(o))))
            if key in seen:
                continue
            seen.add(key)
            picked.append(int(o))
        return picked

    def _candidate_offset_mapping_pairs(self, variables):
        """Propose (offset, mapping) pairs using cheap proxies only (no Plan objects)."""
        base = self.curr_plan.get_actions()
        identity = {v: v for v in variables}
        top_k = self.search_params.get("top_k", 10)
        refine_iters = self.search_params.get("refine_iters", 2)

        pairs = {}
        for o in self.get_top_offsets(base):
            # Identity mapping at good offsets = the old offset-only search
            pairs[(o, tuple(identity.items()))] = self.score_all_shifts(base)[o]

            # Alternate: best mapping for offset -> best offset for mapping -> ...
            mapping = self.get_best_mapping(o, base, variables)
            for _ in range(refine_iters):
                o = int(np.argmax(self.score_all_shifts(self.swap_variables(mapping))))
                mapping = self.get_best_mapping(o, base, variables)
            s = self.score_all_shifts(self.swap_variables(mapping))[o]
            pairs[(o, tuple(mapping.items()))] = s

        def n_swaps(items):
            return sum(k != v for k, v in items)

        ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], n_swaps(kv[0][1])))
        return [(o, dict(items)) for (o, items), _ in ranked[:top_k]]

    def search_optimal_internal_swap(self, variables) -> dict:
        optimal_plan = self.curr_plan
        best_score = optimal_plan.get_score()
        optimal_swap, optimal_offset = {}, 0
        evaluated = set()

        for offset, mapping in self._candidate_offset_mapping_pairs(variables):
            actions = self.shift_actions(self.swap_variables(mapping), offset)
            plan_list = format_actions_to_plan(actions)
            plan_key = make_list_hashable(plan_list)
            if plan_key in evaluated:
                continue
            evaluated.add(plan_key)

            plan = Plan(plan_list, self.gt_plan, self.is_valid, self.domain_name, self.instance_id,
                        to_simulate=False, unique_id=f"{self.unique_id}_{uuid.uuid4()}" if self.unique_id else None)
            if not plan.is_valid:
                plan.check_plan_validity()

            if plan.is_valid:
                self.evaluate_valid_plan(plan)
                self.possible_plans.append(plan)
                return {"optimal_plan": plan, "best_score": plan.get_score(), "optimal_mapping": mapping, "optimal_offset": offset}

            self.evaluate_plan(plan)
            self.possible_plans.append(plan)
            score = plan.get_score()
            if score > best_score:
                best_score, optimal_plan = score, plan
                optimal_swap, optimal_offset = mapping, offset
                if plan.get_actions() != self.curr_plan.get_actions():
                    self.consistent_variable_swap = True

        if optimal_plan.plan != self.curr_plan.plan:
            optimal_plan.states = tuple(deepcopy(optimal_plan.simulate_states()))

        return {"optimal_plan": optimal_plan, "best_score": best_score, "optimal_mapping": optimal_swap, "optimal_offset": optimal_offset}

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
        new_actions = [
            [a[0]] + [mapping.get(x, x) for x in a[1:]]
            for a in self.curr_plan.get_actions()
        ]

        return new_actions

    def score_all_shifts(self, plan_actions):
        """
        Score every left circular shift of the plan against the ground truth in one vectorised pass.
        For each ground-truth position: 1 point if the action names match, plus 1 if the whole action
        (name and arguments) matches.

        :param plan_actions: the plan's actions, each as [action_name, arg1, arg2, ...]
        :return: array where entry s is the score of the plan shifted by s
        """
        plan_length, gt_length = len(plan_actions), len(self.gt_actions)
        if plan_length == 0 or gt_length == 0:
            return np.zeros(max(plan_length, 1))

        # IDs for action names and actions
        name_ids, full_action_ids = {}, {}

        def encode(actions):
            names = np.array([name_ids.setdefault(action[0], len(name_ids)) for action in actions])
            full_actions = np.array([full_action_ids.setdefault(tuple(action), len(full_action_ids))
                                     for action in actions])
            return names, full_actions

        plan_names, plan_full_actions = encode(plan_actions)
        gt_names, gt_full_actions = encode(self.gt_actions)

        # position_index[shift, gt_position] = plan position that lines up with gt_position after the shift
        shifts = np.arange(plan_length)[:, None]
        gt_positions = np.arange(gt_length)[None, :]
        position_index = (gt_positions + shifts) % plan_length

        name_matches = (plan_names[position_index] == gt_names).sum(axis=1)
        full_action_matches = (plan_full_actions[position_index] == gt_full_actions).sum(axis=1)
        return name_matches + full_action_matches

    def get_group_types(self, variables):
        groups = defaultdict(list)
        for v in variables:
            groups[self.var_types.get(v.lower(), "?")].append(v)  # unknown marked as "?"
        return groups

    def get_best_mapping(self, shift_amount, plan_actions, plan_variables):
        """
        For a given circular shift, find the renaming of plan variables that makes the shifted plan match the
        ground-truth plan in as many argument slots as possible.

        :param shift_amount: how far the plan is shifted (left circular shift)
        :param plan_actions: the plan's actions
        :param plan_variables: all variables used in the plan
        :return: dict mapping each plan variable to its new name
        """
        variable_index = {variable: idx for idx, variable in enumerate(plan_variables)}
        num_variables = len(plan_variables)
        plan_length = len(plan_actions)

        # agreement_matrix[u, v] = how much evidence there is that plan variable u should be renamed to v.
        agreement_matrix = 1e-3 * np.eye(num_variables) # small preference for identity mapping if there is a tie

        for position, gt_action in enumerate(self.gt_actions):
            plan_action = plan_actions[(position + shift_amount) % plan_length]

            # Compare only actions with the same name
            if plan_action[0] != gt_action[0] or len(plan_action) != len(gt_action):
                continue

            # Each argument slot votes for renaming the plan's variable to the ground truth's variable
            for plan_variable, gt_variable in zip(plan_action[1:], gt_action[1:]):
                if plan_variable in variable_index and gt_variable in variable_index:
                    agreement_matrix[variable_index[plan_variable], variable_index[gt_variable]] += 1

        mapping = {}
        # One small assignment problem per PDDL type
        for variables_of_type in self.get_group_types(plan_variables).values():
            indices = [variable_index[variable] for variable in variables_of_type]
            type_matrix = agreement_matrix[np.ix_(indices, indices)]

            # linear_sum_assignment minimises cost, so negating the agreement to maximise it
            row_indices, column_indices = linear_sum_assignment(-type_matrix)
            for row, column in zip(row_indices, column_indices):
                mapping[variables_of_type[row]] = variables_of_type[column]

        return mapping

    def search_optimal_offset(self, plan: Plan) -> int:
        actions = plan.get_actions()
        optimal_offset, best_score = 0, plan.get_score()
        evaluated_plans = {}

        for offset in self.get_top_offsets(actions):
            shifted = self.shift_actions(actions, offset)
            plan_list = format_actions_to_plan(shifted)
            plan_key = make_list_hashable(plan_list)

            if plan_key in evaluated_plans:
                continue  # already seen an identical plan

            shifted_plan = Plan(plan_list, self.gt_plan, self.is_valid, self.domain_name, self.instance_id,
                                to_simulate=False,
                                unique_id=f"{self.unique_id}_{uuid.uuid4()}" if self.unique_id else None)
            if not shifted_plan.is_valid:
                shifted_plan.check_plan_validity()

            if shifted_plan.is_valid:   
                self.evaluate_valid_plan(shifted_plan)
                return offset
            self.evaluate_plan(shifted_plan)
            self.possible_plans.append(shifted_plan)

            score = shifted_plan.get_score()
            evaluated_plans[plan_key] = score
            if score > best_score:
                best_score, optimal_offset = score, offset
                if offset != 0:
                    self.offset_improval = True

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

    def offset_heuristic(self, actions: list[list[str]], offset: int) -> int:
        """
        Score how well the shifted actions align with the ground truth by comparing action names at each position.
        """
        n = len(actions)
        score = 0
        for i, gt_action in enumerate(self.gt_actions):
            if actions[(i + offset) % n][0] == gt_action[0]:
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
        if len(plan.get_plan()) == 0 or len(self.gt_plan) == 0:
            plan.decrease_score(1.0)
            return

        chosen_len_penalty_weight = self.penalise_length_from_chosen(plan.get_plan())
        # opt_len_penalty_weight = self.penalise_length_from_optimal(plan.get_plan()) # TODO use somehow?

        # Scoring by action correctness and similarity
        action_similarity = plan.get_similarity_score()
        # print(f"Action similarity: {action_similarity}, and length penalty: {chosen_length_score}")

        plan.increase_score(action_similarity)  # WAS - opt_length_score)

        # Reduce score so that new score is current_score * opt_len_penalty_weight, i.e. the score is reduced by the penalty weight
        current_score = plan.get_score()
        plan.decrease_score(current_score * chosen_len_penalty_weight) # TODO double check

        # print( f"DEBUG NORMAL: Current score for plan {plan.get_plan()} is {plan.get_score()} after action similarity of {action_similarity} and length penalty of {chosen_len_penalty_weight}")

    def evaluate_valid_plan(self, plan) -> None:
        """
        Compute score between current plan and ground truth plan based on optimality metrics.
        """
        plan.increase_score(1)  # make initial score max  = 1 # len(self.curr_plan.get_plan())

        self.valid_plan = plan.get_plan()
        opt_len_penalty_weight = math.sqrt(self.penalise_length_from_optimal(plan.get_plan()))  # Square root so less dramatic than invalid plan
        # Reduce score so that new score is current_score * opt_len_penalty_weight, i.e. the score is reduced by the penalty weight
        current_score = plan.get_score()
        plan.decrease_score(current_score * (1 - opt_len_penalty_weight))

    def run_evaluation(self):
        if self.curr_plan is not None and len(self.curr_plan.get_plan()) > 0 and len(self.gt_plan) > 0:
            print(f"Evaluating plan {self.curr_plan.get_plan()} against GT plan {self.gt_plan}")
            if self.is_valid:
                # print(f"Initial score for valid plan: {len(self.curr_plan.get_plan())}")
                self.evaluate_valid_plan(self.curr_plan)  # penalise on cost from a max of 1
                # self.potential_score = 4 * len(self.gt_plan) / len(self.curr_plan.get_plan())  # penalise on length
                self.potential_score = self.curr_plan.get_score() # Same as final score for valid plan
                assert self.potential_score >= 0.0 and round(self.potential_score, 2) <= 1, f"NORMAL: Potential score {self.potential_score} is out of bounds for valid plan of length {len(self.curr_plan.get_plan())}"
                # print(f"Potential score for valid plan: {self.potential_score}")
            else:
                self.evaluate_plan(self.curr_plan)
                # Checks of potential improvement for best plan candidate
                nlp_plan, nlp_score = self.check_consistent_variable_swap()
                # If there is a consistent solution higher than a threshold, set it as the NLP plan and set the potential score to that score
                self.candidate_plan = nlp_plan
                self.potential_score = nlp_score
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
        chooser = ChooseGTPlan(instance[1], [instance[2]])  # TODO change to all possible GT plans
        chooser.choose_gt_plan_heuristic()

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