# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to evaluate aspects of the invalid plan and provide an overall weighted score

import os
import re
from copy import deepcopy
import uuid

from tarski.io import PDDLReader

from utils import parse_actions, read_config, get_config_dir, get_instance_from_config, parse_action
from common import get_largest_subset, get_largest_subsequence, \
    calculate_semantic_similarity, parse_init_state, parse_goal_state

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
            elif "Plan failed to execute" in line:
                break

    return effects


def apply_action(effects, state) -> list[str]:
    """
    Parse the output of the plan execution and apply the action to the state
    :param effects: the effects of the current action
    :param state: the current state to apply the action to
    :return: the new state after applying the action
    """
    new_state = state
    if not effects:
        # print("No effects found for state in plan validation")
        return new_state

    for effect in effects:
        if "adding" in effect:
            new_state.append(effect[1])
        elif "deleting" in effect:
            if effect[1] in new_state:
                new_state.remove(effect[1])
            else:
                print(f"Effect {effect[1]} not in current state")

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

        self.score: float = len(plan)  # initial score is the length of the plan
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
        self.nr_plan_trace: dict[int, str] = {}  # holds index and quality of the action for the non-redundant actions
        self.trace_score: int = 0  # score of the plan trace
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
        if len(self.plan) == len(self.plan_trace):
            self.set_plan_trace_score()
        else:
            self.set_plan_trace_score()
            print(f"Plan trace length: {len(self.plan_trace)} and plan length: {len(self.plan)}")

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

        with open("temp_plan.txt", "w") as file:
            # for line in llm_plan:
            for act in self.plan:
                file.write(act + "\n")

        cmd = f"{VAL_PATH}/validate -v {self.domain_file} {self.problem_file} temp_plan.txt 2>&1 | tee plans/plan_validation_{self.unique_id}.txt"
        response = os.popen(cmd).read()
        if 'Problem in domain' in response:
            self.is_valid = False
        if "Plan valid" in response:
            self.is_valid = True
        else:
            self.is_valid = False

    def analyse_plan_admissibility(self) -> None:
        """
        Get from the plan validation the action from which the plan is inadmissible (preconditions not met)
         - action > the action from which the plan is inadmissible
         - executable > whether the plan is executable (can be executable but invalid)
        """
        act = []
        next_is_act = False
        executable = False

        # Check if the plan is admissible
        with open(f"plans/plan_validation_{self.unique_id}.txt", "r") as f:
            for line in f:
                if next_is_act:
                    act.append(line.strip())
                    break
                if "Plan failed because of unsatisfied precondition in:" in line:
                    next_is_act = True
                elif "Plan executed successfully" in line:
                    executable = True
                elif "Plan failed to execute" in line: # No action is admissible
                    executable = False
                    act = []
                    break

        self.analysis = {"action": act, "executable": executable}

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
        plan_size = None

        # Plan is partially executable
        if final_action:
            try:
                plan_size = self.plan.index(final_action[0])
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
        # Subset is preferred (consecutive actions)
        subsequence_score = len(self.largest_subsequence) # WAS / len(self.gt_plan)  # largest subsequence
        subset_score = 2 * len(self.largest_subset) # WAS / len(self.gt_plan)  # largest subset (consecutive actions)

        # Scoring by action and variable similarity
        curr_plan, gt_plan, position_score = self.check_action_positioning()  # correct action in correct position
        curr_plan, gt_plan, correctness_score = self.check_action_correctness(curr_plan, gt_plan)  # correct action but wrong position
        similarity_score = self.check_action_similarity(curr_plan, gt_plan)  # correct action but wrong variables

        self.check_similarity_redundant_actions()  # check quality of redundant actions

        return position_score + correctness_score + similarity_score + subsequence_score + subset_score

    def check_action_positioning(self) -> (list[list[str]], list[list[str]], float):
        """
        Check for every action in the list, whether it is identical to the action in the ground truth in this position.
        """
        gt_plan = deepcopy(self.gt_plan)
        curr_plan = deepcopy(self.plan)
        pairs = []  # the actions from gt_plan that have been compared to successfully

        in_position = 0
        i = 0

        while i < len(curr_plan):
            action = curr_plan[i]
            # action and its position are correct
            if i < len(self.gt_plan):
                if action in self.gt_plan and self.gt_plan[i] == action:
                    in_position += 1  # action is in the same position as in ground truth
                    self.action_pairs.append((action, action, "position"))
                    self.plan_trace[i] = "position"  # add action index to plan trace
                    pairs.append(action)
            i += 1

        # Remove all paired actions from the actions lists
        for a in pairs:
            try:
                gt_plan.remove(a)
                curr_plan.remove(a)
            except ValueError:
                print(f"ValueError: checking for action positioning, for action {a} in gt_plan {gt_plan} and gt_pairs {pairs}")

        return curr_plan, gt_plan, in_position

    def check_action_correctness(self, curr_actions, gt_actions) -> (list[str], list[str], float):
        """
        For every action in the list that is not identical to the gt action in its position, check whether it is
        identical to some action in the ground truth that has not been paired yet.
        :return: lists of actions in LLM plan and GT plan, that have not been paired to any other.
        """
        correct = 0
        pairs = []  # the actions from gt_plan that have been compared to successfully
        new_actions = deepcopy(curr_actions)  # the actions from curr_actions that have not been compared to successfully
        comparable_gt_actions = deepcopy(gt_actions)

        i = 0
        # Don't want to account for multiple occurrences of the same action in the current plan
        while i < len(self.plan) and len(self.plan_trace) < len(self.gt_plan):
            action = self.plan[i]
            # action is correct and in the same position
            if self.plan_trace.get(i):
                i += 1
                continue
            # action is correct but position is not, and it had not yet been accounted for
            elif action in comparable_gt_actions:
                correct += 1
                trace = self.plan_trace.get(i, False)
                if not trace:
                    # add action index to plan trace only if position was wrong
                    parsed_act = parse_action(action)
                    self.action_pairs.append((parsed_act, parsed_act, "correct"))
                    self.plan_trace[i] = "correct"
                    pairs.append(action)
                    comparable_gt_actions.remove(action)
                    new_actions.remove(action)
            i += 1

        # Remove all paired actions from the actions lists (once)
        for a in pairs:
            # Only remove if the action is in both lists, otherwise it exists more times in the current plan than in the GT plan
            if a in curr_actions and a in gt_actions:
                curr_actions.remove(a)
                gt_actions.remove(a)

        assert curr_actions == new_actions
        assert gt_actions == comparable_gt_actions

        # action which is correct but not in position adds (up to) a 1/3 of the score of that which is
        score = 0.5 * correct

        return curr_actions, gt_actions, score

    def check_action_similarity(self, curr_actions, gt_actions) -> float:
        """
        Compute similarity of actions in the current plan which are not in the GT plan
        - Match actions with the same act (regardless of variables)
        - Then compare the variables (binary)
        :param curr_actions: all actions in the current plan which are not in the GT plan
        :param gt_actions: all actions in the GT plan which are not in the current plan
        """
        score = 0
        action_pairs = self.get_action_pairs(curr_actions, gt_actions)
        if len(gt_actions) > 0:
            score += len(action_pairs) / 2 # WAS (2 * len(gt_actions))  # give half score for successful action pairs (have a matching action)
        for action_score in action_pairs:
            score += action_score # WAS / len(self.gt_plan)

        return score

    def get_action_pairs(self, curr_actions: list[str], gt_actions: list[str]) -> list[float]:
        """
        For each action in the remaining actions, get the similarity score for the GT action it is most similar to.
        Takes positioning into account, as actions closer to the correct position are more likely to be correct.
        :param curr_actions: all actions in the current plan which are not in the GT plan
        :param gt_actions: all actions in the GT plan which are not in the current plan
        :return: a dictionary of actions from the generated plan and their highest similarity score
        """
        action_similarity = []
        gt_actions_to_pair: list[list[str]] = parse_actions(deepcopy(gt_actions))
        if curr_actions and gt_actions:
            for i in range(len(curr_actions)):
                action = curr_actions[i]
                parsed_action = parse_action(action)
                max_similarity = (0, "redundant") # similarity, corresponding state
                matching_action = []
                state = "redundant"
                act = parsed_action[0]
                og_ind = self.plan.index(action, i)  # index not from reduced list
                # If the action is already paired, find the next occurrence
                while self.plan_trace.get(og_ind):
                    try:
                        og_ind = self.plan.index(action, og_ind + 1)
                    except ValueError:
                        print(f"ValueError: action {action} was found last at index {og_ind}, "
                              f"which already has a state {self.plan_trace.get(og_ind)}")
                        break
                j = 0
                # No more action in the GT plan to pair with
                if len(self.action_pairs) >= len(self.gt_plan):
                    if not self.plan_trace.get(og_ind):
                        self.plan_trace[og_ind] = "redundant"
                    else:
                        next_ind = self.plan.index(action, i + 1)
                        if next_ind and not self.plan_trace.get(next_ind):
                            self.plan_trace[next_ind] = "redundant"
                    continue
                # Compare the action with all remaining actions in the GT plan
                while j < len(gt_actions_to_pair):
                    a = gt_actions_to_pair[j]
                    if a[0] == act:
                        worse_priority = ["diff_act", "redundant"]
                        curr_state = self.plan_trace.get(og_ind)
                        if not curr_state or curr_state in worse_priority:
                            state = "same_act"
                        similarity = 1  # same act in the action (max similarity)
                        similarity += self.compare_variables(parsed_action, a)
                    else:
                        if not self.plan_trace.get(og_ind):
                            state = "diff_act"
                        similarity = calculate_semantic_similarity(act, a[0])  # Wu-Palmer similarity
                        # print(f"Similarity between {act} and {a[0]}: {similarity}")
                        similarity += self.compare_variables(parsed_action, a)
                    if similarity > max_similarity[0]:
                        max_similarity = (similarity, state)
                        matching_action = a
                    j += 1
                if matching_action: # i.e. max_similarity[0] > 0
                    action_similarity.append(max_similarity[0])
                    # remove the matched action from the list of options, and add the pair to the list
                    gt_actions_to_pair.remove(matching_action)
                    best_state = max_similarity[1]
                    self.action_pairs.append((parsed_action, matching_action, best_state))
                    # if action has some similarity, mark it as such
                    self.plan_trace[og_ind] = best_state

        for i in range(len(self.actions)):
            # If any action in the current plan is not matching to the GT plan, mark as redundant
            if not self.plan_trace.get(i):
                self.plan_trace[i] = "redundant"

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

        gt_act_names = [act[0] for act in self.gt_actions] # get the names of the actions in the GT plan

        for i in redundant:
            if self.plan[i] in self.gt_plan:
                # Get all the indices of redundant actions which would have been correct
                # (no need to check position because they couldn't have been more than the GT)
                self.nr_plan_trace[i] = "correct"
                # print(f"Action {self.plan[i]} is redundant but would have been correct")
            elif self.actions[i][0] in gt_act_names:
                # Get all the indices of redundant actions which could match an action name in the GT plan
                self.nr_plan_trace[i] = "same_act"
                # print(f"Action {self.plan[i]} is redundant but has the same act as an action in the GT plan")
            else:
                similarity = 0
                j = 0
                # Check if there is an action in the GT plan which is somehwat similar to the redundant action
                while similarity == 0 and j < len(self.gt_plan):
                    # print(f"Comparing {self.plan[i]} with {self.gt_plan[j]}")
                    action = self.actions[i]
                    a = self.gt_actions[j]
                    similarity = self.compare_variables(action, a)
                    j += 1
                # If there is some similarity, mark the action as diff_act
                if similarity > 0:
                    # print(f"Action {self.plan[i]} is redundant but has some similarity to an action in the GT plan")
                    self.nr_plan_trace[i] = "diff_act"

    def compare_variables(self, action, a) -> float:
        """
        Compare the variables of 2 actions.
        :param action: action from current plan
        :param a: action from ground truth
        :return: the level of similarity between the actions
        """
        score = 0  # penalisation of errors
        similarity = 0  # normalised similarity between the variables

        if len(action) < 2:
            return score

        vars_action = action[1:]
        vars_gt = a[1:]

        if len(vars_action) == len(vars_gt):
            for v1, v2 in zip(vars_action, vars_gt):
                # If the variables are the same in the same position, add 0.25 to the similarity
                if v1 == v2:
                    similarity += 0.25
                # If the variables are the same but in different positions, add 0.1
                elif v1 in vars_gt:
                    similarity += 0.1
        else:
            # Penalty for mismatched number of parameters
            arity_diff = abs(len(vars_action) - len(vars_gt))
            score -= 0.1 * arity_diff

            matches = 0
            for v in vars_action:
                if v in vars_gt:
                    matches += 1
            # If all variables are the same (regardless of position), add 0.75 to the similarity
            if matches == len(vars_action):
                similarity += 0.75
            elif matches > 0:
                similarity += 0.25 * matches

        return similarity + score

    def set_plan_trace_score(self):
        """
        Set the score of the plan trace
        """
        values = {
            "position": 4,
            "correct": 2,
            "same_act": 1,
            "diff_act": 0.5,
            "redundant": 0
        }
        for value in self.plan_trace.values():
            self.trace_score += values.get(value)

    def set_steps_to_validity(self, steps: list):
        self.steps_to_validity = steps

    def get_ind_last_exec(self) -> int:
        """
        Get the index of the last executable action in the plan
        :return: the index of the last executable action in the plan
        """
        act = self.analysis.get("action")
        if len(act) != 0:
            # TODO should probably add index in the parsing of plan validation to get the index of the action
            return self.plan.index(act[0]) # The index would be the number of executable actions
        else:
            # If the plan is empty, or failed to execute: return 0
            if len(self.plan) == 0 or self.analysis.get("executable") is False:
                return 0
            # If the plan is not empty and is executable but no last act, then it is fully executable: return the length of the plan
            return len(self.actions)


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

    def get_score(self):
        """
        Get the score of the plan normalised by the length of the plan
        :return:
        """
        if len(self.plan) == 0:
            return 0
        return self.score / len(self.plan)

    def get_trace_score(self):
        """
        Get the score of the plan trace
        :return:
        """
        if len(self.plan) == 0:
            return 0
        return self.trace_score / len(self.plan)