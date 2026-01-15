# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Utility functions to import and export plans from LLM-planning-analysis and PlanBench

import json
import csv
import os
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))

def parse_plan(plan: str) -> list[str]:
    """
    parse the processed plan from PlanBench into a list which is a sequence of actions as strings.
    :return: a list of actions as strings
    """
    # Handle failed generation of plans by LLM or planner (FD)
    if plan is None:
        return []

    plan = plan.split('\n')
    parsed_plan = []
    for action in plan:
        if len(action) == 0:
            continue
        else:
            parsed_plan.append(action)
    return parsed_plan

def unparse_plan(plan: list[str]) -> str:
    """
    Unparse the plan from a list of actions as strings to a string
    :param plan:
    :return:
    """
    return "\n".join(plan)

def parse_actions(plan: list[str]) ->list[list[str]]:
    """
    parse the plan which is a list of actions as strings to a list of actions.
    :param plan: the parsed plan
    :return: a list of actions as lists of strings
    """
    actions = []
    for action in plan:
        action = action.replace("(", "").replace(")", "")
        if len(action) == 0:
            continue
        actions.append(action.split(' '))

    return actions

def parse_action(action: str) -> list[str]:
    """
    Parse the action from a string to a list of strings
    :param action:
    :return:
    """
    action = action.replace("(", "").replace(")", "")
    if len(action) == 0:
        return []
    return action.split(' ')

def unparse_actions(actions: list[list[str]]) -> list[str]:
    """
    Unparse the actions from a list of actions as lists of strings to a list of actions as strings
    :param actions: the parsed plan
    :return: a list of actions as strings
    """
    plan = []
    for action in actions:
        plan.append(f"({action[0]} {' '.join(action[1:])})")

    return plan

def format_actions_to_plan(actions: list[list[str]]) -> list[str]:
    """
    Parse the actions from a list of lists of strings to a plan where each action is a string, for easier similarity checks
    :param actions: a list of actions which compose a plan
    :return: a list of actions as strings
    """
    plan = []

    for action in actions:
        new_action = f"({action[0]} {' '.join(action[1:])})"
        plan.append(new_action)

    return plan

def get_plans_from_json(filepath) -> list[tuple[int, str, str, bool]]:
    """
    Return a list with tuples of curr_plan and gt_plan from the results JSON file
    :param filepath: path to the results JSON file
    :return: list of tuples (instance_id, curr_plan, gt_plan)
    """
    plans = []
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
        for instance in data["instances"]:
            try:
                plans.append((instance["instance_id"], instance["extracted_llm_plan"], instance["ground_truth_plan"], instance["llm_correct"]))
            except KeyError:
                plans.append((instance["instance_id"], None, None, None))
                print(f"KeyError in instance {instance['instance_id']}")

    return plans

def get_plans_validity_from_json(filepath) -> list[tuple[int, bool]]:
    """
    Return a list with instance id and boolean values from the results JSON file
    :param filepath: path to the results JSON file
    :return: list of tuples (instance_id, validity)
    """
    plans_validity = []
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
        for instance in data["instances"]:
            try:
                plans_validity.append((instance["instance_id"], instance["llm_correct"]))
            except KeyError:
                plans_validity.append((instance["instance_id"], None))
                print(f"KeyError in instance {instance['instance_id']}")

    return plans_validity

def set_configuration(model:str, task="", domain="blocksworld_3", project="llm_planning_analysis", temp=0.1, top_p=1) -> dict[str, str]:
    """
    :param model: model used for plans generation (qwen, llama, qcode, llama_instruct)
    :param task: the task run, see README of project for details ("" -> "_one_shot", "_state_tracking", "_pddl", "_zero_shot")
    :param domain: PDDL domain (blocksworld_3, logistics...)
    :param project: llm_planning_analysis / plan-bench
    :param temp: temperature for the model (default 0.1)
    :param top_p: top-p parameter for the model (default 1)
    :return:
    """
    if len(task) == 0:
        task = "_one_shot"

    return {"project": project, "domain": domain, "model": model, "task": task, "temperature": temp, "top_p": top_p}

def append_evaluation_to_csv(output_file, instance_id, plan, result, validity, parameters):
    """
    :param output_file: Path to the output file.
    :param instance_id: The ID of the instance.
    :param plan: info related to the LLM generated plan, including gt_plan, curr_plan, plan_trace, nr_plan_trace, trace_score, complementary_plan, correct_part, executable, final_score
    :param result: info related to the eval metric results of all plans, including curr_score, curr_steps, curr_last_exec, cdt_score, cdt_steps, cdt_last_exec, subseq_score, subseq_steps, subseq_last_exec
    :param parameters: Dictionary containing the experiment parameters (model, domain, task, temperature, top_p).
    :param validity: info related to the validity of the plans, including curr_validity, cdt_validity, subseq_validity, improved_validity, improved_cdt_validity, improved_subseq_validity, improved_subseq_cdt_validity
    """

    model = parameters.get("model", "")
    domain = parameters.get("domain", "")
    task = parameters.get("task", "")
    temperature = parameters.get("temperature", "")
    top_p = parameters.get("top_p", "")
    header = ["model", "domain", "task", "instance_id", "gt_plan", "curr_plan", "plan_trace", "nr_plan_trace", "trace_score", "complementary_plan", "correct_part", "executability",
              "final_score", "curr_score", "curr_steps", "curr_last_act", "curr_validity", "cdt_plan", "cdt_score", "cdt_steps", "cdt_last_act", "cdt_validity", "subseq_plan", "subseq_score", "subseq_steps", "subseq_last_act", "subseq_validity", "cdt_subseq_plan", "cdt_subseq_score", "cdt_subseq_steps", "cdt_subseq_last_act", "cdt_subseq_validity",
              "improved_plan", "improved_validity"]

    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)

        row = [
            model, domain, task, instance_id,
            plan.get("gt_plan", ""), plan.get("curr_plan", ""), plan.get("plan_trace", ""), plan.get("nr_plan_trace", ""),
            plan.get("trace_score", -100), plan.get("complementary_plan", ""), plan.get("correct_part", ""),
            plan.get("executable", None), plan.get("final_score", -100),
            result.get("curr_score", -100), result.get("curr_steps", -100), result.get("curr_last_exec", -100), validity.get("curr_validity", None),
            plan.get("cdt_plan", ""), result.get("cdt_score", -100), result.get("cdt_steps", -100), result.get("cdt_last_exec", -100), validity.get("cdt_validity", None),
            plan.get("subseq_plan", ""), result.get("subseq_score", -100), result.get("subseq_steps", -100), result.get("subseq_last_exec", -100), validity.get("subseq_validity", None),
            plan.get("cdt_subseq_plan", ""), result.get("cdt_subseq_score", -100), result.get("cdt_subseq_steps", -100), result.get("cdt_subseq_last_exec", -100), validity.get("cdt_subseq_validity", None),
            result.get("improved_plan", ""), validity.get("improved_validity", None),
        ]
        writer.writerow(row)

def read_config(config_file):
    """
    Read the configuration file for the domain; function from llm_planning_analysis/response_evaluation.py
    :param config_file: path to the YAML configuration file
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_config_dir(config_name):
    """
    Get the domain PDDL file from the project
    :param config_name: the name of the domain configuration
    :return: the path to the domain directory
    """
    domain_dirs = {
        "blocksworld": "blocksworld",
        "blocksworld_3": "blocksworld",
        "mystery_blocksworld_3": "blocksworld/mystery",
        "random_blocksworld_3": "blocksworld",
        "logistics": "logistics",
        "obfuscated_randomized_logistics": "obfuscated_randomized_logistics",
        "office_robot": "office_robot"
    }

    return os.path.join(parent_dir, f"LLM-Planning-PlanBench/llm_planning_analysis/instances/{domain_dirs[config_name]}")

def get_instance_from_config(config_name, domain_path, inst_num) -> str:
    """
    Get path to instance file from its directory
    :param config_name: the name of the domain configuration
    :param domain_path: the path to the domain directory
    :param inst_num: the id of the instance
    :return: the path to the specifies instance file
    """
    instance_dirs = {
        "blocksworld": "generated_basic",
        "blocksworld_3": "generated_basic_3",
        "mystery_blocksworld_3": "generated_basic_3",
        "random_blocksworld_3": "generated_basic_3",
        "logistics": "generated_basic",
        "obfuscated_randomized_logistics": "generated_basic",
        "office_robot": "generated_basic"
    }

    return os.path.join(domain_path, os.path.join(instance_dirs[config_name], f"instance-{inst_num}.pddl"))

def get_default_values_for_csv():
    return {
        'plan_trace': '{}',
        'nr_plan_trace': '{}',
        'trace_score': 0.0,
        'complementary_plan': '[]',
        'correct_part': '[]',
        'executability': False,
        'final_score': 0.0,
        'curr_score': 0.0,
        'curr_steps': 10.0,
        'curr_last_act': 0,
        'curr_validity': False,
        'cdt_plan': '[]',
        'cdt_score': 0.0,
        'cdt_steps': 0.0,
        'cdt_last_act': 0,
        'cdt_validity': False,
        'subseq_plan': '[]',
        'subseq_score': 0.0,
        'subseq_steps': 10.0,
        'subseq_last_act': 0,
        'subseq_validity': False,
        'cdt_subseq_plan': '[]',
        'cdt_subseq_score': 0.0,
        'cdt_subseq_steps': 10.0,
        'cdt_subseq_last_act': 0,
        'cdt_subseq_validity': False,
        'improved_plan': '[]',
        'improved_validity': False,
        'improved_cdt_plan': '[]',
        'improved_cdt_validity': False,
        'improved_subseq_plan': '[]',
        'improved_subseq_validity': False,
        'improved_subseq_cdt_plan': '[]',
        'improved_subseq_cdt_validity': False,
    }
