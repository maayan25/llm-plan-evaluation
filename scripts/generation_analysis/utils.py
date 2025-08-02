
import csv
import os

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
    exec_df = fix_executability(df)
    dedup_df = deduplicate_results(exec_df)
    label_df = change_label_names(dedup_df)
    norm_df = normalise_scores(label_df, global_df)
    # norm_df = label_df.copy()

    # Add steps column for improved plan - if plan length is equal to GT, set steps to 0, otherwise set to the absolute differencein lengths
    norm_df["improved_steps"] = norm_df.apply(
        lambda x: 0 if len(ast.literal_eval(x["improved_plan"])) == len(ast.literal_eval(x["gt_plan"]))
        else (abs(len(ast.literal_eval(x["improved_plan"])) - len(ast.literal_eval(x["gt_plan"])))), axis=1)

    return norm_df

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

def combine_csv_results(output_file, results_dir, result_filter = "") -> None:
    """
    Combine the results from the evaluation of generated plans into a single csv file
    :param output_file: the path to the output csv file
    :param results_dir: the directory where the results are stored
    :param result_filter: the filter to apply to the results (e.g. model, domain, task)
    """
    results_dir = results_dir + "o4-mini/" if "o4-mini" in result_filter else results_dir
    result_filter = result_filter.replace("o4-mini", "") if "o4-mini" in result_filter else result_filter
    files = os.listdir(results_dir)

    if len(files) == 0:
        print("No csv files were found in this directory")
        return

    combined_data = []
    all_headers = set()
    file_data = []

    # Filter out directories
    files = [file for file in files if os.path.isfile(os.path.join(results_dir, file))]

    # Filter and read files based on the result_filter
    if result_filter: # Assuming filter is the domain
        files = [file for file in files if file.startswith(result_filter) and file.endswith(".csv")]
    else:
        files = [file for file in files if file.endswith(".csv") and not file.startswith("full_evaluation_")]

    for file in files:
        with open(os.path.join(results_dir, file), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
            if not data:
                continue
            all_headers.update(reader.fieldnames)
            file_data.append(data)

    # Create a consistent ordered header list
    all_headers = sorted(all_headers)

    # Collect all rows, aligned to the combined header
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
    df["improved_validity"] = df.apply(lambda x: 1 if x["improved_plan"] == x["gt_plan"] else x["improved_validity"],
                                       axis=1)

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
    prompt_filename = os.path.join(llm_project_dir, "prompts", domain, "task_1_plan_generation.json")
    if not os.path.exists(prompt_filename):
        raise FileNotFoundError(f"Prompt file for domain '{domain}' does not exist: {prompt_filename}")
    with open(prompt_filename, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
        for instance in prompt_data['instances']:
            if instance['instance_id'] == instance_id:
                gt_plan = instance['ground_truth_plan']
                break
        if not gt_plan:
            gt_plan = ""

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
