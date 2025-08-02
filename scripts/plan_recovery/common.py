# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# File for common functions used in the plan recovery scripts

from utils import parse_plan
from nltk.corpus import wordnet as wn


def get_largest_subsequence(curr_plan, gt_plan) -> list[str]:
    """
    Get the largest subsequence of the ground truth plan that is in the current plan, from the gt plan that is closest
    to the current plan.
    A subsequence can be achieved by removing some or no elements from the lists without changing the order of the
    remaining elements.
    If the subsequence is of length of the GT plan, then potentially no regeneration is required.
    :param curr_plan: the generated plan to evaluate
    :param gt_plan: the ground truth plan to compare against
    :return: the size of the largest subsequence
    """
    # TODO should check against other GT plans as well?
    if curr_plan == gt_plan:
        return curr_plan

    # Space optimised dynamic programming, based on https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/
    m = len(curr_plan)
    n = len(gt_plan)
    matr = [[0] * (n + 1) for _ in range(m + 1)]
    subset = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if curr_plan[i - 1] == gt_plan[j - 1]:
                matr[i][j] = matr[i - 1][j - 1] + 1
                subset[i][j] = subset[i - 1][j - 1] + "\n" + curr_plan[i - 1]  # same structure for str as in PlanBench output...
            else:
                # matr[i][j] = max(matr[i - 1][j], matr[i][j - 1])
                if matr[i - 1][j] > matr[i][j - 1]:
                    matr[i][j] = matr[i - 1][j]
                    subset[i][j] = subset[i - 1][j]
                else:
                    matr[i][j] = matr[i][j - 1]
                    subset[i][j] = subset[i][j - 1]

    subseq = subset[m][n]
    # print(f"Subsequence computed: {subseq}, type: {type(subseq)}")
    return parse_plan(subseq)


def get_largest_subset(curr_plan, gt_plan) -> list[str]:
    """
    Get the largest subset of the ground truth plan that is in the current plan, from the gt plan that is closest
    to the current plan.
    A subset is only a contiguous sequence of elements from the list.
    :param curr_plan: the generated plan to evaluate
    :param gt_plan: the ground truth plan to compare against
    :return: the size of the largest subset
    """
    # TODO Check if it includes the goal or init state?
    if curr_plan == gt_plan:
        return curr_plan

    # Space optimised dynamic programming, based on https://www.geeksforgeeks.org/longest-common-substring-dp-29/
    m = len(curr_plan)
    n = len(gt_plan)
    matr = [[0] * (n + 1) for _ in range(m + 1)]

    longest_sub = 0
    end = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if curr_plan[i - 1] == gt_plan[j - 1]:
                matr[i][j] = matr[i - 1][j - 1] + 1
                if matr[i][j] > longest_sub:
                    longest_sub = matr[i][j]
                    end = i
            else:
                matr[i][j] = 0  # reset to 0

    start = end - longest_sub
    return curr_plan[start:end]

def check_special_states(curr_plan, gt_plan) -> tuple[float, float]:
    """
    Check if the plan includes the goal state, the init state, or both.
    :param curr_plan: the generated plan to evaluate
    :param gt_plan: the ground truth plan to compare against
    :return: a tuple of floats, where 0 is not existing, 0.5 is wrong position, and 1 is correct -> (init, goal)
    """
    init = 0
    goal = 0

    # Cases of failed generation
    if len(curr_plan) == 0 or len(gt_plan) == 0:
        return init, goal

    if curr_plan[0] == gt_plan[0]:
        init = 1
    elif curr_plan[0] in gt_plan:
        init = 0.5

    if curr_plan[-1] == gt_plan[-1]:
        goal = 1
    elif curr_plan[-1] in gt_plan:
        goal = 0.5

    return init, goal

def calculate_semantic_similarity(act1, act2):
    """
    Calculate the semantic similarity between two actions names
    :param act1:
    :param act2:
    :return:
    """
    syn1 = wn.synsets(act1, pos=wn.VERB)
    syn2 = wn.synsets(act2, pos=wn.VERB)

    if len(syn1) == 0 or len(syn2) == 0:
        return 0.0

    return syn1[0].wup_similarity(syn2[0])

def parse_goal_state(problem):
    """
    Parse the goal state from the problem
    :param problem: the instance parsed with PDDLReader
    :return: the goal state as a list of atoms
    """
    # Check if goal is surrounded by parentheses
    goal = str(problem.goal)
    if goal.startswith("(") and goal.endswith(")"):
        goal = goal[1:-1].strip()
    parsed_goal = []

    predicates = goal.split(" and ")
    for pred in predicates:
        pred_name = pred.split("(")[0].strip()
        args_content = pred.split("(")[1].split(")")[0].strip()

        if args_content != "":
            args = " ".join(args_content.split(","))
            new_pred = f"({pred_name} {args})"
        else:
            new_pred = f"({pred_name})"

        parsed_goal.append(new_pred)
    print(f"Parsed goal state: {parsed_goal}")
    return parsed_goal


def parse_init_state(problem):
    """
    Parse the initial state of the problem from the PDDLReader instance
    :param problem: the instance parsed with PDDLReader
    :return: the initial state as a list of strings
    """
    init = problem.init.as_atoms()
    parsed_init = []
    for pred in init:
        pred = str(pred)
        pred_name = pred.split("(")[0]
        args_content = pred.split("(")[1].split(")")[0]

        if args_content != "":
            args = " ".join(args_content.split(","))
            new_pred = f"({pred_name} {args})"
        else:
            new_pred = f"({pred_name})"
        parsed_init.append(new_pred)
    return parsed_init


def get_variable_lists(actions: list[list[str]]) -> list[str]:
    """
    list of all the variables that are used in the given actions
    :param actions: a list of actions
    :return: a list of all the variables which are used in these actions
    """
    variables = []

    for action in actions:
        if action[1] not in variables:
            variables.append(action[1])
        if len(action) > 2 and action[2] not in variables:
            variables.append(action[2])

    return variables

def make_list_hashable(obj):
    if isinstance(obj, list):
        return tuple(make_list_hashable(e) for e in obj)
    return obj
