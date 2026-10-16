# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Utility functions to import and export plans from LLM-planning-analysis and PlanBench

import json
import csv
import os
import yaml

import re
from collections import defaultdict
from functools import lru_cache
from types import MappingProxyType

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

def get_plans_from_json(filepath) -> list[dict[str, any]]:
    """
    Return a list with tuples of curr_plan and gt_plan from the results JSON file
    :param filepath: path to the results JSON file
    :return: list of tuples (instance_id, curr_plan, gt_plan)
    """
    plans = []
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
        for instance in data["instances"]:
            plans.append({"instance_id": instance["instance_id"],
                          "extracted_llm_plan": instance.get("extracted_llm_plan", None),
                          "ground_truth_plan": instance.get("ground_truth_plan", None),
                          "llm_correct": instance.get("llm_correct", None)})
    return plans

def get_llm_plans_from_json(filepath) -> dict[int, str]:
    """
    Return a dictionary with instance id and llm plan from the results JSON file
    :param filepath: path to the results JSON file
    :return: dictionary of instance_id and llm_plan
    """
    llm_plans = {}
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
        for instance in data["instances"]:
            try:
                llm_plans[instance["instance_id"]] = instance["extracted_llm_plan"]
            except KeyError:
                llm_plans[instance["instance_id"]] = None
                print(f"KeyError in instance {instance['instance_id']}")

    return llm_plans

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

def get_row_for_csv(instance_id, plan, result, validity, parameters, search_params, search_time) -> list:
    """
    Get the row for the CSV file based on the provided parameters, without writing it to the file.
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
        search_params.get("top_k", None), search_params.get("min_k", None), search_params.get("max_k", None), search_params.get("tol", None), search_params.get("refine_iters", None), search_time
    ]
    return row

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
        "mystery_blocksworld": "blocksworld/mystery",
        "random_blocksworld": "blocksworld",
        "random_blocksworld_3": "blocksworld",
        "logistics": "logistics",
        "obfuscated_randomized_logistics": "obfuscated_randomized_logistics",
        "obfuscated_deceptive_logistics": "obfuscated_deceptive_logistics",
        "office_robot": "office_robot",
        "elevator": "elevator",
        "satellite": "satellite"
    }

    return os.path.join(parent_dir, f"LLM-Planning-PlanBench/llm_planning_analysis/instances/{domain_dirs[config_name]}")

def get_instances_dir(config_name) -> str:
    """
    Return the directory name for the instances of a given domain configuration
    :param config_name: the name of the domain configuration
    :return: the directory name for the instances of the specified domain configuration
    """
    instance_dirs = {
        "blocksworld": "generated_basic",
        "blocksworld_3": "generated_basic_3",
        "mystery_blocksworld_3": "generated_basic_3",
        "random_blocksworld_3": "generated_basic_3",
        "mystery_blocksworld": "generated_basic",
        "random_blocksworld": "generated_basic",
        "logistics": "generated_basic",
        "obfuscated_randomized_logistics": "generated_basic",
        "obfuscated_deceptive_logistics": "generated_basic",
        "office_robot": "generated_basic",
        "elevator": "generated_basic",
        "satellite": "generated_basic"
    }
    return instance_dirs[config_name]

def get_instance_from_config(config_name, domain_path, inst_num) -> str:
    """
    Get path to instance file from its directory
    :param config_name: the name of the domain configuration
    :param domain_path: the path to the domain directory
    :param inst_num: the id of the instance
    :return: the path to the specifies instance file
    """
    instances_dir = get_instances_dir(config_name)
    return os.path.join(domain_path, os.path.join(instances_dir, f"instance-{inst_num}.pddl"))

def get_default_values_for_csv(gt_plan):
    """
    Return the list that should be the row in the CSV file when the plan is not generated
    :return: a csv row with default values for the plan evaluation metrics
    """
    # header = ["gt_plan", "curr_plan", "plan_trace", "nr_plan_trace",
    #           "trace_score", "complementary_plan", "correct_part", "executability",
    #           "final_score", "curr_score", "curr_steps", "curr_last_act",
    #           "curr_validity", "cdt_plan", "cdt_score", "cdt_steps",
    #           "cdt_last_act", "cdt_validity", "subseq_plan", "subseq_score",
    #           "subseq_steps", "subseq_last_act", "subseq_validity", "cdt_subseq_plan",
    #           "cdt_subseq_score", "cdt_subseq_steps", "cdt_subseq_last_act", "cdt_subseq_validity",
    #           "improved_plan", "improved_validity"]
    # If GT plan is not None, set it to the GT plan and all steps are its length, otherwise set it to an empty list and steps to 0.
    if gt_plan is not None:
        gt_plan_list = parse_plan(gt_plan)
        gt_steps = len(gt_plan_list)
    else:
        gt_plan_list = []
        gt_steps = 0

    # All other values floats are 0.0, ints are 0, booleans are False, and lists are empty
    # Floats: trace_score, final_score, curr_score, cdt_score, subseq_score, cdt_subseq_score
    # Ints: instance_id, curr_last_act, cdt_last_act, curr_steps, cdt_steps, subseq_steps, subseq_last_act, cdt_subseq_steps, cdt_subseq_last_act
    # Lists: gt_plan, curr_plan, plan_trace, nr_plan_trace, complementary_plan, correct_part, cdt_plan, subseq_plan, cdt_subseq_plan, improved_plan
    # Booleans: executability, curr_validity, cdt_validity, subseq_validity, cdt_subseq_validity, improved_validity
    row = [gt_plan_list, [], {}, {}, # "gt_plan", "curr_plan", "plan_trace", "nr_plan_trace",
           0.0, gt_plan_list, [], False, # "trace_score", "complementary_plan", "correct_part", "executability",
           0.0, 0.0, gt_steps, 0,  # "final_score", "curr_score", "curr_steps", "curr_last_act",
           False, [], 0.0, gt_steps,  # "curr_validity", "cdt_plan", "cdt_score", "cdt_steps",
           0, False, [], 0.0,  # "cdt_last_act", "cdt_validity", "subseq_plan", "subseq_score",
           gt_steps, 0, False, [], # "subseq_steps", "subseq_last_act", "subseq_validity", "cdt_subseq_plan",
           0.0, gt_steps, 0, False, # "cdt_subseq_score", "cdt_subseq_steps", "cdt_subseq_last_act", "cdt_subseq_validity",
           gt_plan_list, True] # "improved_plan", "improved_validity"

    return row

def read_file(path):
    with open(path) as f:
        return re.sub(r";.*", "", f.read()).lower()      # strip comments, lowercase

def return_expression(text, start):
    """Return the balanced (...) expression starting at text[start] == '('."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]

@lru_cache(maxsize=None)
def dynamic_predicates(domain_file: str) -> frozenset:
    """Every symbol appearing inside any :effect (predicates that actions can change). Cached per domain."""
    text = read_file(domain_file)
    dyn = set()
    for m in re.finditer(r":effect", text):
        j = text.find("(", m.end())
        if j != -1:
            dyn.update(re.findall(r"\(\s*([^\s()]+)", return_expression(text, j)))
    return frozenset(dyn)

@lru_cache(maxsize=None)
def get_object_types(problem_file: str, domain_file: str | None = None):
    """
    Map each object to a type name.
      1. Declared type in (:objects ... - type)
      2. else the static unary predicate in :init, e.g. (TRUCK t0) -> "truck". If several apply, e.g. LOCATION + AIRPORT,
      keep the one with the most members; assumed to be the supertype.
      3. else "object" (untyped)
    """
    text = read_file(problem_file)
    types = {}

    m = re.search(r"\(:objects", text)
    if m:
        tokens = return_expression(text, m.start())[len("(:objects"):-1].split()
        pending, i = [], 0
        while i < len(tokens):
            if tokens[i] == "-":
                for o in pending:
                    types[o] = tokens[i + 1]
                pending, i = [], i + 2
            else:
                pending.append(tokens[i])
                i += 1
        for o in pending:
            types[o] = "object"

    m = re.search(r"\(:init", text)
    if m and domain_file:
        dyn = dynamic_predicates(domain_file)
        ext = defaultdict(set)                            # predicate -> objects
        for pred, obj in re.findall(r"\(\s*([^\s()]+)\s+([^\s()]+)\s*\)", return_expression(text, m.start())):
            if pred not in dyn:
                ext[pred].add(obj)
        cands = defaultdict(list)                         # object -> static unary predicates
        for pred, objs in ext.items():
            for o in objs:
                cands[o].append(pred)
        for o, preds in cands.items():
            if types.get(o, "object") == "object":
                types[o] = max(preds, key=lambda p: (len(ext[p]), p))

    return MappingProxyType(types)                        # read-only, safe to share via the cache