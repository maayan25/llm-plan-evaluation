# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to evaluate aspects of the invalid plan and provide an overall weighted score
import json
import os
import re
from copy import deepcopy
import uuid

from tarski.io import PDDLReader

from utils import parse_actions, read_config, get_config_dir, get_instance_from_config, parse_action
from common import get_largest_subset, get_largest_subsequence, \
    calculate_semantic_similarity, parse_init_state, parse_goal_state, is_meaningful_word

import subprocess
import tempfile

# Run the file from its directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = os.path.abspath(os.path.join(project_dir, os.pardir))

llm_project_dir = os.path.join(parent_dir, "LLM-Planning-PlanBench/")

def get_effects(unique_id) -> dict[int, list[str]]:
    """
    Get the effects of the action from the plan validation output file
    :param unique_id: the unique id of the plan
    :return: a dictionary of action index and its effects
    """
    effects = {}
    i = 1
    with open(f"plans/plan_validation_{unique_id}.txt", "r") as f:
        if not f:
            print("No plan validation file found")
            return effects
        for line in f:
            if "Checking next happening" in line:
                effects[i] = []
                i += 1
            match = re.match(r"(Deleting|Adding) \((.+)\)", line)
            if match:
                effect_type = match.group(1)  # 'Deleting' or 'Adding'
                effect = [effect_type.lower(), f"({match.group(2)})"]
                effects[i - 1].append(effect)
                # Check for deleting the "calibrated" predicate
                deleting_calibrated = effect_type == "Deleting" and match.group(2).startswith("calibrated")
                if deleting_calibrated:
                    print(f"DEBUG: Deleting calibrated predicate in action {i - 1} for plan {unique_id}")
            elif "Plan failed to execute" in line:
                break

    return effects


def apply_action(effects, state) -> list[str]:
    """
    :param effects: the effects of the current action
    :param state: the current state to apply the action to
    :return: the new state after applying the action
    Apply an action's effects to a state, following PDDL semantics: deletes first, then adds.
    """
    new_state = list(dict.fromkeys(state))  # copy, drop duplicate facts, keep order
    if not effects:
        # print("No effects found for state in plan validation")
        return new_state

    for effect in effects:
        if effect[0] == "deleting":
            if effect[1] in new_state:
                new_state.remove(effect[1])
            else:
                print(f"[Info] Effect {effect[1]} not in current state")
                # print(f"Current state: {new_state}\n Effects: {effects}")

    for effect in effects:
        if effect[0] == "adding" and effect[1] not in new_state:
            new_state.append(effect[1])

    return new_state

class Plan:
    def __init__(self, plan: list[str], gt: list[str], valid=False, domain="blocksworld_3", instance_id=2, to_simulate=True, unique_id=None):
        self.plan: list[str] = plan
        self.actions: list[list[str]] = parse_actions(plan)
        self.gt_plan:list[str] = gt
        self.gt_actions: list[list[str]] = parse_actions(gt)
        if unique_id is None:
            self.unique_id = str(uuid.uuid4())
        else:
            self.unique_id = unique_id

        self.score: float = 0  # initial score is 0
        self.is_valid: bool = valid

        self.config_name = domain
        self.instance_id = instance_id

        self.config_file = f"{llm_project_dir}/llm_planning_analysis/configs/{self.config_name}.yaml"
        self.data = read_config(self.config_file)

        self.domain_path = get_config_dir(self.config_name)
        self.domain_file = f"{self.domain_path}/generated_domain.pddl"
        self.problem_file = get_instance_from_config(self.config_name, self.domain_path, self.instance_id)

        self.domain = None  # domain parsed with PDDLReader
        self.problem = None  # instance parsed with PDDLReader
        self.set_domain_problem()  # set domain and problem from parsed PDDL files

        self.action_pairs = []  # holds the action pairs
        self.plan_trace: dict[int, str] = {}  # holds index and quality of the action
        self.action_similarity: dict[int, float] = {} # holds index and similarity score of the action
        self.nr_plan_trace: dict[int, str] = {}  # holds index and quality of the action for the non-redundant actions
        self.analysis = {"action": [], "executable": True}

        # Check if plan is a suboptimal valid plan with a validator, and analyse the plan admissibility
        # TODO change this for GT plans?
        self.check_plan_validity()
        self.analyse_plan_admissibility()
        # print(f"Plan is valid: {self.is_valid} and analysis: {self.analysis}")

        self.largest_subsequence = get_largest_subsequence(self.plan, self.gt_plan)
        self.largest_subset = get_largest_subset(self.plan, self.gt_plan)

        # Simulate the states that the plan will go through
        self.to_simulate = to_simulate
        self.simulation_possible = True
        self.states = None  # holds the states that the plan goes through

        self.similarity = self.evaluate_similarity()
        if self.to_simulate and not self.states:
            simulated = self.simulate_states()
            self.states = tuple(deepcopy(simulated))
            # print(f"DEBUG SIM: STATES after assignment: {self.states}")

        # print(f"plan_trace: {self.plan_trace},\n for plan {self.plan}\n and GT plan {self.gt_plan}")
        self.steps_to_validity: list = []
        if not len(self.plan) == len(self.plan_trace):
            print(f"Warning: plan trace length {len(self.plan_trace)} and plan length {len(self.plan)}")
        if not len(self.plan) == len(self.action_similarity):
            print(f"Warning: action similarity length {len(self.action_similarity)} and plan length {len(self.plan)}")
        self.trace_score: float = self.set_plan_trace_score()  # score of the plan trace

    def set_domain_problem(self) -> None:
        """
        Set the domain and problem from the PDDL files
        :return: None
        """
        reader = PDDLReader(raise_on_error=True)
        self.domain = reader.parse_domain(self.domain_file)
        self.problem = reader.parse_instance(self.problem_file)

    def check_plan_validity(self) -> None:
        """
        Check if the plan is valid using VAL validator
        """
        VAL_PATH = os.path.join(llm_project_dir, "planner_tools/VAL")
        domain_file_abs = os.path.abspath(self.domain_file)
        problem_file_abs = os.path.abspath(self.problem_file)
        validate_bin = os.path.join(VAL_PATH, "validate")

        os.makedirs("plans", exist_ok=True)
        validation_log_path = f"plans/plan_validation_{self.unique_id}.txt"

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_plan_path = os.path.join(tmpdir, "temp_plan.txt")
            with open(temp_plan_path, "w") as file:
                for act in self.plan:
                    file.write(act + "\n")

            result = subprocess.run(
                [validate_bin, "-v", domain_file_abs, problem_file_abs, temp_plan_path],
                capture_output=True, text=True
            )
            response = result.stdout + result.stderr

        with open(validation_log_path, "w") as f:
            f.write(response)

        self.is_valid = "Plan valid" in response

    import re

    def analyse_plan_admissibility(self) -> None:
        """
        Get from the plan validation the action from which the plan is inadmissible (preconditions not met)
         - action > the action from which the plan is inadmissible
         - action_index > the 0-indexed position of that action in the plan (None if not found/applicable)
         - executable > whether the plan is executable (can be executable but invalid)
        """
        act = []
        next_is_act = False
        executable = False
        last_time_step = None
        time_pattern = re.compile(r"Checking next happening \(time (\d+)\)")

        with open(f"plans/plan_validation_{self.unique_id}.txt", "r") as f:
            for line in f:
                if next_is_act:
                    act.append(line.strip())
                    break

                time_match = time_pattern.search(line)
                if time_match:
                    # print(f"DEBUG: Found time step {time_match.group(1)} in plan validation for plan {self.unique_id}")
                    last_time_step = int(time_match.group(1))
                elif "Plan failed because of unsatisfied precondition in:" in line:
                    next_is_act = True
                elif "Plan executed successfully" in line:
                    executable = True
                elif "Plan failed to execute" in line:  # No action is admissible
                    executable = False
                    act = []
                    last_time_step = None
                    break

        # VAL's "time" is 1-indexed and increments once per action checked; last one printed before the failure should be the failing action.
        action_index = int(last_time_step - 1) if (act and last_time_step is not None) else None

        self.analysis = {"action": act, "action_index": action_index, "executable": executable}

    def simulate_states(self) -> list[list[str]]:
        """
        Simulate the states that the plan will go through by applying the actions in the plan to the initial state,
        :return: a list of all the states that the plan goes through
        """
        # Rerun the plan validation to get the admissibility of the concerned plan
        self.check_plan_validity()
        self.analyse_plan_admissibility()

        init = parse_init_state(self.problem)
        states = [deepcopy(init)]
        current_state = deepcopy(init)

        final_action = self.analysis.get("action")
        executable = self.analysis.get("executable")
        final_action_index = self.analysis.get("action_index")
        plan_size = None

        # Plan is partially executable
        if final_action:
            try:
                # plan_size = self.plan.index(final_action[0])
                plan_size = final_action_index
            except ValueError as e: # For some reason failed in Logistics, even when the action is identical to the one in the plan
                print(f"ValueError: action {final_action[0]} not found in plan, simulation cannot be completed")
                self.simulation_possible = False
                self.analysis["executable"] = False
                self.analysis["action"] = []
                return states
        else:
            if executable: # Plan is fully executable
                plan_size = len(self.actions)
            # else Plan is not executable
            else:
                plan_size = 0 # Aligns with the "no action is admissible" case

        # If retrieving from the plan fails, get from the plan validation file
        if self.config_name == "logistics":
            if plan_size is None:
                with open(f"plans/plan_validation_{self.unique_id}.txt", "r") as f:
                    for line in f:
                        if "Plan size" in line:
                            plan_size = int(line.split(": ")[1])
                    if not plan_size:
                        # Should not ever happen
                        print("Plan size not found in plan validation file, simulation cannot be completed")
                        self.simulation_possible = False
                        return states

        if plan_size == 0:
            return states

        effects = get_effects(self.unique_id)
        if not effects:
            print("Plan validation has failed, simulation cannot be completed")
            self.simulation_possible = False
            return states

        # Apply all effects of actions up to the last action
        for i in range(1, plan_size + 1):
            # print(f"DEBUG SIM: Applying action {i}, with effects: {effects.get(i)} for instance {self.instance_id}")
            new_state = apply_action(effects.get(i), current_state)
            states.append(deepcopy(new_state))
            current_state = deepcopy(new_state)
            # print(f"DEBUG SIM: New state: {new_state}")

        # print(f"DEBUG SIM: states are {states} for instance {self.instance_id}")
        return deepcopy(states)

    def simulate_states_backwards(self) -> list[list[str]]:
        """
        Simulate the states that the plan will go through by de-applying the actions in the plan to the goal state,
        :return: a list of all the states that the plan goes through in reverse
        """
        # TODO implement in the future?
        goal = parse_goal_state(self.problem)
        states = [deepcopy(goal)]
        current_state = deepcopy(goal)

        return states

    def evaluate_similarity(self):
        if len(self.plan) == 0:
            return 0.0

        # Subset is preferred (consecutive actions)
        subsequence_score = len(self.largest_subsequence) # WAS / len(self.gt_plan)  # largest subsequence
        subset_score = len(self.largest_subset) # WAS / len(self.gt_plan)  # largest subset (consecutive actions)

        # Scoring by action and variable similarity
        gt_plan, gt_indices, position_score = self.check_action_positioning()  # correct action in correct position
        curr_plan, gt_plan, gt_indices, correctness_score = self.check_action_correctness(gt_plan, gt_indices)  # correct action but wrong position
        similarity_score = self.check_action_similarity(curr_plan, gt_plan, gt_indices)  # correct action but wrong variables

        self.check_similarity_redundant_actions()  # check quality of redundant actions for non-redundant plan trace (NP-AQM)

        # score = (position_score + correctness_score + similarity_score + subsequence_score + 2 * subset_score) / 4 # Divided by 4 because a perfect plan will get a score which is 4 times the plan length
        lcs_score = (1/3 * subsequence_score) + (2/3 * subset_score)
        score = 0.5 * (position_score + correctness_score + similarity_score) + 0.5 * lcs_score  # weighted score of the actions in the plan and the largest subsequence/subset

        # print(f"DEBUG: Similarity score components for plan: position: {position_score}, correctness: {correctness_score}, similarity: {similarity_score}, subsequence: {subsequence_score}, subset: {subset_score}, lcs_score: {lcs_score}, final score: {score}")
        # print(f"DEBUG: Accumuative similarity is: {position_score + correctness_score + similarity_score} and lcs score is: {lcs_score}")

        # Score should not be more than the length of the plan, as it is a weighted score of the actions in the plan (1 for correct action in correct position)
        assert round(score, 2) <= len(self.plan), print(f"NORMAL: Similarity score is more than plan length. Score: {score}, plan length: {len(self.plan)} "
                                              f"\n Score components: position: {position_score}, correctness: {correctness_score}, similarity: {similarity_score}, subsequence: {subsequence_score}, subset: {subset_score}")

        return score / len(self.plan)  # normalise score to be between 0 and 1

    def check_action_positioning(self) -> (list[str], list[str], list[int], int):
        """
        Check for every action in the plan whether it is identical to the GT action in the same position.
        :return: the unpaired actions of the LLM plan, the unpaired actions of the GT plan,
                 their original GT indices, and the number of in-position actions.
        """
        in_position = 0
        paired_idx = set()

        for i, action in enumerate(self.plan):
            if i < len(self.gt_plan) and self.gt_plan[i] == action:
                in_position += 1
                self.action_pairs.append((action, action, "position"))
                self.plan_trace[i] = "position"
                self.action_similarity[i] = 1.0
                paired_idx.add(i)

        # Rebuild remaining actions by index
        # curr_plan = [a for i, a in enumerate(self.plan) if i not in paired_idx]
        gt_indices = [k for k in range(len(self.gt_plan)) if k not in paired_idx]
        gt_plan = [self.gt_plan[k] for k in gt_indices]

        return gt_plan, gt_indices, in_position

    def check_action_correctness(self, gt_actions, gt_indices) -> (list[str], list[str], float):
        """
        For every action in the list that is not identical to the gt action in its position, check whether it is
        identical to some action in the ground truth that has not been paired yet.
        :return: lists of actions in LLM plan and GT plan, that have not been paired to any other.
        """
        gt_pool = list(zip(gt_indices, gt_actions))  # (gt_idx, action) together
        correct = 0

        for i, action in enumerate(self.plan):
            if len(self.plan_trace) >= len(self.gt_plan):
                break
            if self.plan_trace.get(i):
                continue
            match_pos = next((k for k, (_, a) in enumerate(gt_pool) if a == action), None)
            if match_pos is None:
                continue
            gt_pool.pop(match_pos)
            correct += 0.5
            parsed_act = parse_action(action)
            self.action_pairs.append((parsed_act, parsed_act, "correct"))
            self.plan_trace[i] = "correct"
            self.action_similarity[i] = 0.5

        curr_actions = [a for i, a in enumerate(self.plan) if not self.plan_trace.get(i)]
        gt_indices = [k for k, _ in gt_pool]
        gt_actions = [a for _, a in gt_pool]
        return curr_actions, gt_actions, gt_indices, correct  # adjust to your caller

    def get_indices_from_plan_trace(self) -> dict[str, int]:
        """
        Get the indices of the actions that will be checked for similarity, by checking which indices in the plan trace are not already paired with a GT action.
        :return: the indices of the actions that will be checked for similarity
        """
        indices_to_check = {}
        index = 0
        for i in range(len(self.plan)):
            if not self.plan_trace.get(i):
                indices_to_check[str(index)] = i
                index += 1
        return indices_to_check

    def check_action_similarity(self, curr_actions, gt_actions, gt_indices) -> float:
        """
        Compute similarity of actions in the current plan which are not in the GT plan
        - Match actions with the same act (regardless of variables)
        - Then compare the variables (binary)
        :param curr_actions: all actions in the current plan which are not in the GT plan
        :param gt_actions: all actions in the GT plan which are not in the current plan
        """
        score = 0
        action_pairs = self.get_action_pairs(curr_actions, gt_actions, gt_indices=gt_indices)
        for action_score in action_pairs:
            score += action_score # WAS / len(self.gt_plan)

        return score

    def get_action_pairs(self, curr_actions: list[str], gt_actions: list[str], gt_indices) -> list[float]:
        """
        For each action in the remaining actions, get the similarity score for the GT action it is most similar to.
        Takes positioning into account, as actions closer to the correct position are more likely to be correct.
        :param curr_actions: all actions in the current plan which are not in the GT plan
        :param gt_actions: all actions in the GT plan which are not in the current plan
        :return: a dictionary of actions from the generated plan and their highest similarity score
        """
        indices_to_check: dict[str, int] = self.get_indices_from_plan_trace()
        action_similarity = []
        gt_actions_to_pair: list[list[str]] = parse_actions(deepcopy(gt_actions))
        gt_indices_to_pair = list(gt_indices)

        if curr_actions and gt_actions:
            for same_action_name in (True, False):
                for i, action in enumerate(curr_actions):
                    parsed_action = parse_action(action)
                    max_similarity = (0, "redundant")  # similarity, corresponding state
                    matching_action = []
                    match_pos = None
                    act = parsed_action[0]
                    og_ind = indices_to_check[str(i)]
                    j = 0

                    # Already paired
                    if self.plan_trace.get(og_ind):
                        continue
                    # No more action in the GT plan to pair with, marked as redundant later
                    if len(self.action_pairs) >= len(self.gt_plan):
                        continue

                    # Compare the action with the remaining GT actions
                    while j < len(gt_actions_to_pair):
                        a = gt_actions_to_pair[j]
                        if (a[0] == act) != same_action_name:
                            j += 1
                            continue
                        if same_action_name:
                            state = "same_act"
                            similarity = (1 + self.evaluate_parameters(parsed_action, a)) / 2
                        else:
                            state = "diff_act"
                            similarity = calculate_semantic_similarity(act, a[0])  # Wu-Palmer similarity
                            similarity += self.evaluate_parameters(parsed_action, a)
                            similarity = similarity / 2
                        in_position = (gt_indices_to_pair[j] == og_ind)
                        similarity = similarity if in_position else similarity * 0.5
                        if similarity > max_similarity[0]:
                            max_similarity = (similarity, state)
                            matching_action = a
                            match_pos = j
                        j += 1

                    if matching_action:
                        max_sim = max_similarity[0]
                        action_similarity.append(max_sim)
                        self.action_similarity[og_ind] = max_sim
                        gt_actions_to_pair.pop(match_pos)
                        gt_indices_to_pair.pop(match_pos)
                        self.action_pairs.append((parsed_action, matching_action, max_similarity[1]))
                        self.plan_trace[og_ind] = max_similarity[1]

        for i in range(len(self.actions)):
            # If any action in the current plan is not matching to the GT plan, mark as redundant
            if not self.plan_trace.get(i):
                self.plan_trace[i] = "redundant"
                self.action_similarity[i] = 0.0

        return action_similarity

    def check_similarity_redundant_actions(self):
        """
        Set the actions that are redundant in the plan trace to the corresponding state, according to the length
        of the GT plan and the level of redundancy (better scoring actions are prioritised)
        :return:
        """
        self.nr_plan_trace = deepcopy(self.plan_trace)

        # Get all the indices of redundant actions
        redundant = [i for i, state in self.plan_trace.items() if state == "redundant"]
        if len(redundant) == 0:
            return

        gt_act_names = {act[0] for act in self.gt_actions}  # get the names of the actions in the GT plan

        for i in redundant:
            action = self.actions[i]
            if self.plan[i] in self.gt_plan:
                self.nr_plan_trace[i] = "correct"
            if action[0] in gt_act_names:
                self.nr_plan_trace[i] = "same_act"
            else:
                if any((calculate_semantic_similarity(action[0], a[0]) + self.evaluate_parameters(action, a)) / 2 > 0
                       for a in self.gt_actions):
                    self.nr_plan_trace[i] = "diff_act"

    def evaluate_parameters(self, action, a) -> float:
        """
        Evaluate the parameters of an action compared to a GT action, by checking whether they are the same or not.
        If parameters are words, compute the semantic similarity between the parameters, and give a score accordingly.
        Otherwise, if they are variables, give a score of 1 if they are the same and 0 otherwise.
        :param action: action from current plan
        :param a: action from ground truth
        :return: 1 if the parameters are the same, 0 otherwise
        """
        if len(action) < 2 and len(a) < 2:
            return 1.0  # both have no params: perfect match
        if len(action) < 2 or len(a) < 2:
            return 0.0  # one has params, the other doesn't

        params_action = action[1:]
        params_gt = a[1:]

        if not params_action or not params_gt:
            return 0.0

        # Check if most parameters are meaningful words (not variables) using wordnet
        meaningful_action = sum(1 for p in params_action if is_meaningful_word(p))
        meaningful_gt = sum(1 for p in params_gt if is_meaningful_word(p))

        # If most parameters are meaningful words, compute semantic similarity between parameters by index, giving more weight to parameters in the correct position
        if meaningful_action > len(params_action) / 2 and meaningful_gt > len(params_gt) / 2:
            similarity = 0
            for i in range(min(len(params_action), len(params_gt))):
                if params_action[i] == params_gt[i]:
                    similarity += 1
                else:
                    similarity += calculate_semantic_similarity(params_action[i], params_gt[i])
            return similarity / max(len(params_action), len(params_gt))
        else:
            return self.compare_variables(action, a)

    def compare_variables(self, action, a, displaced_weight: float = 0.5) -> float:
        """
        Compare the variables of 2 actions.
        :param action: action from current plan
        :param a: action from ground truth
        :param displaced_weight: credit given for a variable that appears in the GT but at the wrong position.
        1.0 = position irrelevant, 0.0 = position is everything. Default 0.5 is a neutral prior which can be tuned
        if labelled data is available per domain.
        :return: the level of similarity between the actions [0, 1]
        """
        params_action = action[1:]
        params_gt = a[1:]

        n_action = len(params_action)
        n_gt = len(params_gt)

        if n_action == 0 and n_gt == 0:
            return 1.0
        if n_action == 0 or n_gt == 0:
            return 0.0

        # Score each parameter in params_action against params_gt
        exact_matches = 0
        displaced_matches = 0
        gt_unmatched = list(params_gt)
        exact_positions = set()

        for i, v in enumerate(params_action):
            if i < n_gt and v == params_gt[i]:
                exact_matches += 1
                exact_positions.add(i)
                gt_unmatched.remove(v)

        for i, v in enumerate(params_action):
            if i in exact_positions:
                continue
            if v in gt_unmatched:
                displaced_matches += 1
                gt_unmatched.remove(v)

        raw_score = exact_matches + displaced_weight * displaced_matches
        # Normalise against the larger of the two lists to penalise arity mismatch
        similarity = raw_score / max(n_action, n_gt)

        return round(similarity, 4)

    def set_plan_trace_score(self):
        """
        Set the score of the plan trace
        """
        if len(self.plan) == 0:
            return 0

        trace_score = 0
        values = {
            "position": 1,
            "correct": 0.5,
            "same_act": 0.25,
            "diff_act": 0.125,
            "redundant": 0
        }
        for value in self.plan_trace.values():
            trace_score += values.get(value)
        return trace_score / len(self.plan)

    def set_steps_to_validity(self, steps: list):
        self.steps_to_validity = steps

    def get_ind_last_exec(self) -> float:
        """
        Get the index of the last executable action in the plan
        :return: the index of the last executable action in the plan
        """
        n = len(self.plan)
        if n == 0:
            return 0.0

        fail_idx = self.analysis.get("action_index") # This is the index of the action that failed to execute, if any.
        act = self.analysis.get("action") or []

        # This shouldn't happen
        if fail_idx is None and act:
            print(f"DEBUG: no failing-action index found. Plan: {self.plan}\n Analysis: {self.analysis}")
            fail_idx = self.plan.index(act[0])

        if fail_idx is not None:
            # print(f"DEBUG: Plan {self.plan} failed at action {fail_idx} ({act}) for instance {self.instance_id}; last executable action position: {fail_idx}")
            # print(f"DEBUG: LEA set to {fail_idx / n}, n = {n}")
            # print(f"DEBUG: number of states simulated: {len(self.states)}")
            assert fail_idx == len(self.states) - 1, print(f"DEBUG: number of states simulated {len(self.states)} does not match the last executable action index {fail_idx} for plan {self.plan} in instance {self.instance_id}")
            return fail_idx / n
        if self.analysis.get("executable") is False:
            # print(f"DEBUG: Plan {self.plan} is not executable for instance {self.instance_id}; LEA set to 0.0")
            return 0.0
        # print(f"DEBUG: Plan {self.plan} is fully executable for instance {self.instance_id}; LEA set to 1.0")
        return 1.0  # fully executable

    def get_plan(self):
        return self.plan

    def get_actions(self):
        return self.actions

    def get_plan_trace(self):
        return self.plan_trace

    def increase_score(self, score):
        self.score += score

    def decrease_score(self, score):
        self.score -= score

    def get_similarity_score(self):
        return self.similarity

    def get_score(self) -> float:
        """
        Get the score of the plan normalised by the length of the plan
        :return:
        """
        if len(self.plan) == 0:
            return 0
        return round(self.score, 4) # / len(self.plan)

    def get_trace_score(self) -> float:
        """
        Get the score of the plan trace
        :return:
        """
        if len(self.plan) == 0:
            return 0
        # final_score = self.trace_score # / len(self.plan)
        assert round(self.trace_score, 2) <= 1.0, print(f"NORMAL: Plan trace score is more than 1. Score: {self.trace_score}, plan length: {len(self.plan)}")
        return round(self.trace_score, 4)

    def json_dump_plan_state(self, filepath, extra=None):
        """Save all instance attributes (and optional extra values) to JSON."""
        data = dict(self.__dict__)
        if extra:
            data.update(extra)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)