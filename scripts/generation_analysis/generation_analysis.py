import csv
import json
import os
import argparse

import re
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Please run the file from its directory or modify the paths
current_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = os.path.abspath(os.path.join(project_dir, os.pardir))

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def plot_query_length_distributions(filenames, labels=None):
    """
    Given a list of JSON filenames, reads the files and extracts the number of words
    in each 'query' field under 'instances', and plots the KDE of these lengths.

    :param filenames: List of JSON filenames to read.
    :param labels: list of labels for each file, used in the plot legend. If not provided, files will be labeled as "File i"
    """
    if labels and len(labels) != len(filenames):
        raise ValueError("Number of labels must match number of filenames")

    all_lengths = []
    for idx, filename in enumerate(filenames):
        # Check if the file exists
        assert os.path.exists(filename), f"File {filename} does not exist"
        print(f"Reading file {idx+1}/{len(filenames)} which is {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            # Load the JSON data
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from file {filename}: {e}")

        lengths = [len(instance['query'].split()) for instance in data.get('instances', [])]
        label = labels[idx] if labels else f"File {idx + 1}"
        all_lengths.append((lengths, label))

    # Plotting
    plt.figure(figsize=(10, 5))
    for lengths, label in all_lengths:
        sns.kdeplot(lengths, label=label, fill=True, linewidth=2)

    # plt.title("Distribution of Query Lengths (in words)")
    plt.xlabel("Number of Words", fontsize=26)
    plt.ylabel("Density", fontsize=26)
    plt.legend(fontsize=18, loc="best")
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{figures_path}/query_length_distribution_{len(filenames)}_types.pdf", format="pdf", dpi="figure")
    plt.show()

def count_goals_in_pddl_files(directory):
    """
    Parses PDDL files to count number of atomic goals in an (:goal (and ...)) block.
    Handles nested parentheses and closing brackets on same lines.
    """
    goal_counts = Counter()

    for filename in os.listdir(directory):
        if not filename.endswith(".pddl") or "_amended" in filename:
            continue

        filepath = os.path.join(directory, filename)
        num_goals = get_goal_counts_for_instance(filepath)

        goal_counts[num_goals] += 1

    return goal_counts

def get_goal_counts_for_instance(instance_path):
    """
    Parses a single PDDL file to count the number of atomic goals in an (:goal (and ...)) block.
    Handles nested parentheses and closing brackets on same lines.
    """
    with open(instance_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r";.*", "", content)
    goal_start = content.find("(:goal")
    if goal_start == -1:
        return 0

    # Track parentheses to extract full goal block
    parens = 0
    goal_block = ""
    inside = False
    for char in content[goal_start:]:
        if char == "(":
            parens += 1
            inside = True
        elif char == ")":
            parens -= 1
        goal_block += char
        if inside and parens == 0:
            break

    # Extract atomic goals inside (and ...)
    match = re.search(r"\(\s*and\s+(.*)\)", goal_block, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1)
        atomic_goals = re.findall(r"\([^\(\)]+\)", inner)
        return len(atomic_goals)

    return 0

def get_goal_counts_for_domain(instances_dir):
    """
    Prints the goal counts in a formatted way.
    """
    goal_counts = count_goals_in_pddl_files(instances_dir)
    print(f"Number of goals in PDDL files for {domain} domain:")
    for num_goals, count in sorted(goal_counts.items()):
        print(f"{num_goals} goals: {count} files")

def extract_goals_to_csv(pddl_dir, output_csv_path):
    """
    Scans PDDL files in the directory and writes a CSV with columns:
    domain, instance_id, num_goals.
    Assumes filenames are of the form instance-N.pddl.
    """
    rows = []

    if not domain:
        combine_domain_goal_counts(output_csv_path)

    for filename in os.listdir(pddl_dir):
        if not filename.endswith(".pddl") or "_amended" in filename:
            continue

        match = re.match(r"instance-(\d+)\.pddl", filename)
        if not match:
            continue
        instance_id = int(match.group(1))
        num_goals = get_goal_counts_for_instance(os.path.join(pddl_dir, filename))

        rows.append({
            "domain": domain,
            "instance_id": instance_id,
            "num_goals": num_goals
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["domain", "instance_id", "num_goals"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv_path}")

def combine_domain_goal_counts(output_csv_path):
    """
    Combines goal counts from multiple domains into a single CSV file.
    Assumes csv files are named as <domain>_goal_counts.csv and located in the same directory as output_csv_path.
    """
    all_rows = []
    output_file = os.path.join(os.path.dirname(output_csv_path), "goal_counts.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["domain", "instance_id", "num_goals"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        all_rows = collect_rows_from_domain_files(output_csv_path, all_rows)
        writer.writerows(all_rows)

def collect_rows_from_domain_files(output_csv_path, all_rows):
    for filename in os.listdir(os.path.dirname(output_csv_path)):
        if not filename.endswith("_goal_counts.csv"):
            continue

        domain = filename.split("_")[0]
        domain_path = os.path.join(os.path.dirname(output_csv_path), filename)

        with open(domain_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row["domain"] = domain
                all_rows.append(row)

        print(f"Collected {len(all_rows)} rows from {filename}")
    return all_rows

def get_instances_dir_from_domain(domain):
    mapping = {
        "": "",
        "logistics": "logistics/generated_basic",
        "blocksworld_3": "blocksworld/generated_basic_3",
        "mystery_blocksworld_3": "blocksworld/mystery/generated_basic_3",
        "random_blocksworld_3": "blocksworld/generated_basic_3",
    }
    return mapping[domain]

def get_query_distributions():
    log_results_dir = f"{results_path}/logistics/temp_{temp}_top_p_{top_p}/{model}"
    bw_results_dir = f"{results_path}/blocksworld_3/temp_{temp}_top_p_{top_p}/{model}"
    plot_query_length_distributions([f"{log_results_dir}/task_1_plan_generation_state_tracking.json",
                                     f"{bw_results_dir}/task_1_plan_generation_state_tracking.json",
                                     f"{log_results_dir}/task_1_plan_generation.json"],
                                    labels=["CoT Logistics", "CoT BW", "oneshot Logistics"]) # , "oneshot BW"
                                    # f"{bw_results_dir}/task_1_plan_generation.json"],

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LLM planning generation results")
    parser.add_argument("--project", type=str, default="llm_planning_analysis", help="Project name")
    parser.add_argument("--model", type=str, default="qwen", help="Model name")
    parser.add_argument("--domain", type=str, default="", help="Domain name")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for LLM generation")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling parameter")

    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    project = args.project
    model = args.model
    domain = args.domain
    temp = args.temp
    top_p = args.top_p

    llm_project_path = os.path.join(parent_dir, "LLM-Planning-PlanBench", project)

    instances_domain = get_instances_dir_from_domain(domain)
    instances_path = os.path.join(llm_project_path, "instances", instances_domain)

    # get_goal_counts_for_domain(instances_path)
    # extract_goals_to_csv(instances_path, f"{project_dir}/results/plan_recovery/{domain}_goal_counts.csv")

    results_path = os.path.join(llm_project_path, "results")
    figures_path = f"{project_dir}/figures/plan_generation/"

    get_query_distributions()