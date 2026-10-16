
import csv
import os
from pathlib import Path

import numpy as np
from scipy import stats as stats
import pandas as pd
import json
import ast

def postprocess_results(df, global_df):
    """
    Postprocess the evaluation results DataFrame by deduplicating and ensuring numeric scores.

    :param df: DataFrame with evaluation results.
    :param global_df: DataFrame with full evaluation results.
    :return: Processed DataFrame with deduplicated entries and numeric scores.
    """
    dedup_df = deduplicate_results(df)
    label_df = change_label_names(dedup_df)
    # norm_df = normalise_scores(label_df, global_df)
    norm_df = denormalise_lea(label_df)
    print(f"Adding optimal plan for {len(norm_df)} rows...")
    w_optimal_plan = add_optimal_plan(norm_df)
    print(f"Optimal plan added for {len(w_optimal_plan)} rows.")
    exec_df = fix_executability(w_optimal_plan)

    # Adds task_name and model_name columns to the results dataframe
    final_df = abbreviate_config_names(exec_df)
    final_df = map_config_in_results(final_df)
    final_df = add_validity_capped_score(final_df)

    # Drop all rows that have missing values
    final_df = final_df.dropna()

    # Add steps column for improved plan - if plan length is equal to optimal GT, set steps to 0, otherwise set to the absolute difference in lengths
    final_df["improved_steps"] = final_df.apply(
        lambda x: 0 if len(ast.literal_eval(x["improved_plan"])) == len(x["optimal_plan"])
        else (abs(len(ast.literal_eval(x["improved_plan"])) - len(x["optimal_plan"]))), axis=1)

    return final_df

def add_validity_capped_score(df, score_col="final_score", validity_col="curr_validity",
                               new_col="final_score_validity"):
    """
    Creates a version of score_col where the value is rescaled based on validity_col:
    - valid (True):   rescaled into [0.5, 1.0]
    - invalid (False): rescaled into [0.0, 0.5]
    """
    df = df.copy()

    if df[validity_col].dtype == object:
        df[validity_col] = df[validity_col].replace({"True": True, "False": False})
    df[validity_col] = df[validity_col].astype(bool)

    clipped_score = df[score_col].clip(0, 1)

    def rescale(row_score, is_valid):
        if is_valid:
            return 0.5 + row_score * 0.5   # [0, 1] -> [0.5, 1.0]
        else:
            return row_score * 0.5          # [0, 1] -> [0.0, 0.5]

    df[new_col] = [
        rescale(score, valid) for score, valid in zip(clipped_score, df[validity_col])
    ]

    return df

def denormalise_lea(df):
    """
    Denormalises the LEA scores in the DataFrame by multiplying them by plan length.
    :param df:
    :return:
    """
    df = df.copy()
    lea_column = "curr_last_act"
    plan_column = "curr_plan"

    # Add plan_length column to the dataframe
    df[plan_column] = df[plan_column].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df["plan_length"] = df[plan_column].apply(len)

    df[lea_column] = df[lea_column].apply(pd.to_numeric, errors="coerce")
    df["plan_length"] = df["plan_length"].apply(pd.to_numeric, errors="coerce")
    df["denormalised_curr_lea"] = df[lea_column] * df["plan_length"]

    return df

def abbreviate_config_names(df):
    model_map = {
        "gpt-3.5-turbo_chat": "GPT-3.5",
        "gpt-4": "GPT-4",
        "o3-mini_chat": "O3",
        "o4-mini_chat": "O4",
        "gemini_2_flash": "Gn2F",
        "gemini_2-5_flash": "Gn2.5F",
        "gemini_2-5_pro": "Gn2.5P",
        "claude_3_haiku": "C3H",
        "claude_3-5_haiku": "C3.5H",
        "claude_3-7_sonnet": "C3.7S",
        "gemma-2-2b_it": "Gm2",
        "qwen": "Q2.5",
        "qcode": "Q2.5C",
        "qwenl": "Q2.57B",
        "qcodel": "Q2.514B",
        "llama": "L3.2",
        "dsqwen": "DSQ",
        "claude-opus-4-8": "COpus4.8",
        "claude-sonnet-4-6": "CSonnet4.6",
        "gemini_3-8_flash": "Gi3.8Flash",
        "gemini_3-1_pro": "Gi3.1Pro",
        "gemma4-12b": "Gm4.12B",
        "gemma4-26b-a4b_it": "Gm4.26B",
        "gpt-6-col_chat": "GPT6Sol",
        "qwen3-8b": "Q3.8B",
        "qwen3-14b": "Q3.14B",
        "qwen3-30b-a3b": "Q3.30B",
    }

    task_map = {
        "_one_shot": "1S",
        "_state_tracking": "CoT",
        "_pddl": "PDDL",
        "_zero_shot": "0S",
    }

    domain_map = {
        "blocksworld": "BW",
        "blocksworld_3": "BW3",
        "elevator": "Elev",
        "satellite": "Sat",
        "mystery_blocksworld": "MBW",
        "random_blocksworld": "RBW",
        "logistics": "Log",
        "obfuscated_deceptive_logistics": "ODL",
        "obfuscated_randomized_logistics": "ORL",
    }

    df["task_name_abbv"] = df["task"].map(task_map).fillna(df["task"])
    df["model_name_abbv"] = df["model"].map(model_map).fillna(df["model"])
    df["domain_name_abbv"] = df["domain"].map(domain_map).fillna(df["domain"])

    return df

def map_config_in_results(df):
    model_map = {
        "gpt-3.5-turbo_chat": "GPT-3.5 Turbo",
        "gpt-4": "GPT-4",
        "o3-mini_chat": "O3 Mini",
        "o4-mini_chat": "O4 Mini",
        "gemini_2_flash": "Gemini 2 Flash",
        "gemini_2-5_flash": "Gemini 2.5 Flash",
        "gemini_2-5_pro": "Gemini 2.5 Pro",
        "claude_3_haiku": "Claude 3 Haiku",
        "claude_3-5_haiku": "Claude 3.5 Haiku",
        "claude_3-7_sonnet": "Claude 3.7 Sonnet",
        "gemma-2-2b_it": "Gemma 2 (2B)",
        "qwen": "Qwen 2.5 (1.5B)",
        "qcode": "Qwen 2.5 Coder (1.5B)",
        "qwenl": "Qwen 2.5 (7B)",
        "qcodel": "Qwen 2.5 Coder (14B)",
        "llama": "Llama 3.2 (3B)",
        "claude-opus-4-8": "Claude Opus 4.8",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "gemini_3-8_flash": "Gemini 3.8 Flash",
        "gemini_3-1_pro": "Gemini 3.1 Pro",
        "gemma4-12b": "Gemma 4 12B",
        "gemma4-26b-a4b_it": "Gemma 4 26B",
        "gpt-6-sol_chat": "GPT 6 Sol",
        "qwen3-8b": "Qwen3 8B",
        "qwen3-14b": "Qwen3 14B",
        "qwen3-30b-a3b": "Qwen3 30B A3B",
        # "dsqwen": "DeepSeek Qwen",
    }

    task_map = {
        "_one_shot": "One-Shot",
        "_state_tracking": "State Tracking",
        "_pddl": "PDDL",
        "_zero_shot": "Zero-Shot",
    }

    domain_map = {
        "blocksworld": "Blocksworld",
        "blocksworld_3": "Blocksworld",
        "elevator": "Elevator",
        "satellite": "Satellite",
        "mystery_blocksworld_3": "Mystery BW",
        "random_blocksworld_3": "Random BW",
        "logistics": "Logistics",
        "obfuscated_randomized_logistics": "Random Logistics",
        "obfuscated_deceptive_logistics": "Obfuscated Deceptive Logistics",
    }

    df["task_name"] = df["task"].map(task_map).fillna(df["task"])
    df["domain_name"] = df["domain"].map(domain_map).fillna(df["domain"])
    df["model_name"] = df["model"].map(model_map).fillna(df["model"])

    return df

def normalise_scores(df, global_df, score_columns=["final_score", "curr_score", "trace_score", "cdt_score", "subseq_score", "cdt_subseq_score"]):
    """
    Normalises the scores in the DataFrame to be in [0, 1] using min-max scaling.
    :param df: DataFrame with evaluation results.
    :param global_df: DataFrame with full evaluation results for reference.
    :param score_columns: List of column names containing scores to normalise.
    :return: DataFrame with normalised scores.
    """
    df = df.copy()

    for score_column in score_columns:
        if score_column in df.columns:
            df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
            min_val = global_df[score_column].quantile(0.1)
            max_val = global_df[score_column].quantile(0.9)
            print(f"Normalising {score_column} with min: {min_val}, max: {max_val}")
            if pd.notna(min_val) and pd.notna(max_val) and max_val != min_val:
                df[score_column] = (df[score_column] - min_val) / (max_val - min_val)
            else:
                df[score_column] = 0
    return df

def deduplicate_results(df, key_columns=["domain", "model", "task", "instance_id"], score_column="final_score"):
    """
    Removes duplicate entries from the DataFrame

    :param df: DataFrame with evaluation results.
    :param key_columns: List of column names to define uniqueness.
    :param score_column: Column name containing the score to maximise.
    :return: Deduplicated DataFrame.
    """
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    dedup_df = df.sort_values(score_column, ascending=False).drop_duplicates(subset=key_columns, keep="first")

    return dedup_df.reset_index(drop=True)

def parse_plan(plan_str):
    """
    Parse a plan string representation into a list of action strings.
    :param plan_str: String representation of the plan (e.g., "['(action1)', '(action2)']")
    :return: List of action strings
    """
    actions = [action.strip() for action in plan_str.split("\n") if action.strip()]
    return actions

def get_optimal_plan(domain, instance_id):
    """
    Get the optimal plan for a given domain and instance_id from the JSON GT plans file.
    :param domain: The domain name (e.g., "logistics")
    :param instance_id: The instance ID to look for
    :return: List of strings representing the optimal plan
    """
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    gt_plans_path = f"{project_dir}/results/gt_plans/{domain}/gt_plans.json"
    with open(gt_plans_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_plans = None
    for instance in gt_data.get("instances", []).keys():
        if int(instance) == instance_id:
            gt_plans = gt_data["instances"][instance]
            gt_plans = [parse_plan(plan_str) for plan_str in gt_plans]

    if gt_plans is not None:
        opt_plan = min(gt_plans, key=len)
        # print(f"Optimal plan for instance_id {instance_id}: {opt_plan}, of length {len(opt_plan)}, and type {type(opt_plan)}")
        return opt_plan
    else:
        raise ValueError(f"Optimal plan for instance_id {instance_id} not found in {gt_plans_path}")

def add_optimal_plan(df):
    """
    Add the optimal plan for each instance in the DataFrame based on the domain and instance_id.
    :param df: the DataFrame containing the results
    :return: the DataFrame with the optimal plan added
    """
    df = df.copy()
    df["optimal_plan"] = df.apply(lambda row: get_optimal_plan(row["domain"], row["instance_id"]), axis=1)
    return df

def combine_csv_results(output_file, results_dir, result_filter="") -> None:
    """
    Combine the results from the evaluation of generated plans into a single csv file.
    Expects results_dir structured as:
        results_dir/<domain>/<model>/evaluation__<task>.csv

    :param output_file: the path to the output csv file
    :param results_dir: the directory where the results are stored
    :param result_filter: the filter to apply to the results (e.g. model, domain, task)
    """
    results_dir = Path(results_dir)
    csv_files = sorted(results_dir.glob("*/*/*.csv"))

    if not csv_files:
        print("No csv files were found in this directory")
        return

    def matches_filter(path: Path) -> bool:
        domain = path.parent.parent.name
        model = path.parent.name
        # filenames look like "evaluation__task.csv" -> extract the task part
        stem = path.stem
        task = stem.split("__", 1)[1] if "__" in stem else stem

        if result_filter:
            return result_filter in (domain, model, task)
        # default (no filter)
        return not path.name.startswith("full_evaluation_")

    files = [f for f in csv_files if matches_filter(f)]

    if not files:
        print("No csv files matched the given filter")
        return

    all_headers = set()
    file_data = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
            if not data:
                continue
            all_headers.update(reader.fieldnames)
            file_data.append(data)

    # Create a consistent ordered header list
    all_headers = sorted(all_headers)

    # Collect all rows, aligned to the combined header
    combined_data = []
    for data in file_data:
        for row in data:
            aligned_row = [row.get(col, "") for col in all_headers]
            combined_data.append(aligned_row)

    # Write to the combined output CSV
    print(f"Writing combined results to {output_file}")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(all_headers)
        writer.writerows(combined_data)


def fix_executability(df):
    """
    Ensure that plans with empty or missing content are marked as not executable.
    Modifies the 'executability' column in-place.
    :param df: the DataFrame to fix
    """
    df = df.copy()

    df["gt_plan"] = df["gt_plan"].apply(lambda x: x if isinstance(x, str) else '[]')
    df["improved_plan"] = df["improved_plan"].apply(lambda x: x if isinstance(x, str) else '[]')
    # Fix validity of improved plans - if improved plan is the same as GT plan, set improved validity to 1
    df["improved_validity"] = df.apply(lambda x: True if (x["improved_plan"] == x["optimal_plan"] or x["improved_plan"] == x["gt_plan"]) else x["improved_validity"], axis=1)

    # Find rows where improved_validity has an invalid value (not True, False, true, false, 1, 0) and set them to False
    invalid_values = df[~df["improved_validity"].isin([True, False, "True", "False", "true", "false", 1, 0])]["improved_validity"].unique()
    if len(invalid_values) > 0:
        print(f"Warning: Found invalid values in 'improved_validity' column: {invalid_values}. Printing key_columns for the rows with invalid values:")
        invalid_rows = df[~df["improved_validity"].isin([True, False, "True", "False", "true", "false", 1, 0])]
        for _, row in invalid_rows.iterrows():
            print(f"Domain: {row['domain']}, Model: {row['model']}, Task: {row['task']}, Instance ID: {row['instance_id']}, Improved Validity: {row['improved_validity']}")

    df["improved_validity"] = df["improved_validity"].replace({"True": 1, "False": 0, "true": 1, "false": 0}).astype(float)

    def is_executable(plan, orig_flag):
        if plan in [None, '', [], '[]', 'nan']:
            return False
        if isinstance(plan, str) and plan.strip() in ['', '[]']:
            return False
        return orig_flag

    df["executability"] = df.apply(lambda row: is_executable(row.get("curr_plan", ""), row["executability"]), axis=1)
    return df


def change_label_names(df, column_name="plan_trace"):
    """
    Change the names of the labels in the plan_trace according to the provided logic.
    :param df: DataFrame containing the results.
    :param column_name: Name of the column containing the plan_trace data.
    :return: DataFrame with updated label names
    """
    df = df.copy()
    df[column_name] = df[column_name].apply(transform_dict)
    return df

def transform_dict(dict_data, column_name="plan_trace"):
    if isinstance(dict_data, str):
        dict_data = eval(dict_data)
    if isinstance(dict_data, dict):
        new_dict = {}
        for index, value in dict_data.items():
            if value == "position":
                new_dict[index] = "correct"
            elif value == "correct" and "nr" not in column_name:
                new_dict[index] = "misplaced"
            else:
                new_dict[index] = value
        return new_dict
    return dict_data

def fisher_r_to_z(r1, r2, n1, n2):
    """
    Perform Fisher's r-to-z transformation test to compare two correlation coefficients.

    :param r1: First correlation coefficient (e.g., score and steps)
    :param r2: Second correlation coefficient (e.g., success and steps)
    :param n1: Sample size for the first correlation
    :param n2: Sample size for the second correlation
    :return: Test statistic and p-value
    """

    # Apply Fisher's r-to-z transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Standard error of the difference between the two z-scores
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

    # Compute the z-difference
    z_diff = (z1 - z2) / se

    # Calculate the p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_diff)))  # Two-tailed test

    return z_diff, p_value


def get_default_values_for_csv(domain, instance_id):
    """
    Returns a dictionary with default values for the CSV file.
    """
    # Get length of GT plan for this domain and instance
    if domain == "office_robot":
        gt_length = 20  # Approximate average length for office_robot
    else:
        gt_plan = get_gt_plan_for_domain_instance(domain, instance_id)
        # Convert from string to list
        gt_plan = gt_plan.split("\n") if gt_plan else []
        gt_length = len(gt_plan)

    return {
        'gt_plan': '[]',
        'curr_plan': '[]',
        'plan_trace': '{}',
        'nr_plan_trace': '{}',
        'trace_score': 0.0,
        'complementary_plan': '[]',
        'correct_part': '[]',
        'executability': False,
        'final_score': 0.0,
        'curr_score': 0.0,
        'curr_steps': gt_length * 2,
        'curr_last_act': 0,
        'curr_validity': False,
        'cdt_plan': '[]',
        'cdt_score': 0.0,
        'cdt_steps': gt_length * 2,
        'cdt_last_act': 0,
        'cdt_validity': False,
        'subseq_plan': '[]',
        'subseq_score': 0.0,
        'subseq_steps': gt_length * 2,
        'subseq_last_act': 0,
        'subseq_validity': False,
        'cdt_subseq_plan': '[]',
        'cdt_subseq_score': 0.0,
        'cdt_subseq_steps': gt_length * 2,
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

def get_gt_plan_for_domain_instance(domain, instance_id):
    """
    Returns the length of the GT plan for the given domain and instance.
    """
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    llm_project_dir = os.path.join(project_dir, os.pardir, "LLM-Planning-PlanBench", "llm_planning_analysis")
    prompt_filename = os.path.join(llm_project_dir, "prompts", domain, "task_1_plan_generation_pddl.json")
    gt_plan = ""
    if not os.path.exists(prompt_filename):
        raise FileNotFoundError(f"Prompt file for domain '{domain}' does not exist: {prompt_filename}")
    with open(prompt_filename, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
        for instance in prompt_data['instances']:
            if instance['instance_id'] == instance_id:
                gt_plan = instance['ground_truth_plan']
                break
        if not gt_plan:
            print(f"Warning: GT plan for domain '{domain}' and instance '{instance_id}' not found in prompt file.")

    return gt_plan

def fill_missing_instances(results, output_file):
    """
    Check for missing instance numbers in each unique (model, task, domain) experiment
    and add rows for them using provided default values.

    :param results: DataFrame containing the results.
    :param output_file: Path to the output CSV file.
    """
    df = results.copy()

    required_instances = set(range(2, 102))
    grouped = df.groupby(['model', 'task', 'domain'])

    missing_rows = []

    for (model, task, domain), group in grouped:
        existing_instances = set(group['instance_id'])
        missing_instances = required_instances - existing_instances

        for instance in missing_instances:
            default_values = get_default_values_for_csv(domain, instance)
            row = {
                'model': model,
                'task': task,
                'domain': domain,
                'instance_id': instance,
                **default_values
            }
            missing_rows.append(row)

    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        df = pd.concat([df, missing_df], ignore_index=True)

    df.sort_values(by=['model', 'task', 'domain', 'instance_id'], inplace=True)
    df.to_csv(output_file, index=False)

    print(f"Missing instances filled and saved to: {output_file}")
