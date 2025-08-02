# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Script to run the evaluation results of generated plans
import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import spearmanr, kruskal, chi2_contingency, kendalltau
from matplotlib.lines import Line2D

from utils import fill_missing_instances, combine_csv_results, postprocess_results

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

class AnalyseResults:
    """
    Class to analyse the results of the evaluation of generated plans
    """
    def __init__(self, results_file, results_dir):
        self.results_file = results_file
        self.results = pd.read_csv(results_file)
        self.global_results = pd.read_csv(f"{results_dir}/full_evaluation.csv")
        self.results = postprocess_results(self.results, self.global_results)

        # Split the results by success rate
        self.success_columns = ["curr_validity", "cdt_validity", "subseq_validity", "cdt_subseq_validity", "improved_validity"]
        if "o4-mini" in args.domain:
            self.high_success, self.low_success = self.split_results_by_success_rate("curr_validity", group_by="domain")
        else:
            self.high_success, self.low_success = self.split_results_by_success_rate("curr_validity")

        # Run all the plots
        if success_level:
            self.results = self.high_success if success_level == "high_success" else self.low_success

        self.run_plots(self.results)
        self.get_analysis(self.results)

    def get_analysis(self, df):
        # Get the average success rates for each plan type
        success_means = {col: df[col].mean() for col in self.success_columns}

        # Get the average steps for each plan type
        step_columns = [col.replace("validity", "steps") for col in self.success_columns]
        step_means = {col: df[col].mean() for col in step_columns}

        # Save the analysis to a csv file
        plans = ["pi_0", "pi_1", "pi_2", "pi_3", "pi_4"]
        analysis_df = pd.DataFrame({
            "Plan Type": plans,
            "Success Rate": [f"{success_means[col]:.2%}" for col in self.success_columns],
            "Average Steps": [f"{step_means[col]:.2f}" for col in step_columns]
        })

        analysis_df.to_csv(f"{tables_dir}/analysis_results.csv", index=False)

        if "blocksworld" in domain:
            self.extract_success_rates_at_lengths(df, "gt_plan", "curr_validity",
                                                  "cdt_subseq_validity")  # DONE TABLE for BW
        elif domain == "logistics":
            self.extract_success_rates_at_lengths(df, "gt_plan", "curr_validity", "cdt_subseq_validity",
                                                  [3, 4, 5, 6, 7, 8, 9, 10, 11])  # DONE TABLE for Logistics

        # Mean plan length for Logistics CoT VS Logistics one-shot
        df["curr_plan"] = df["curr_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["curr_plan_length"] = df["curr_plan"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        mean_plan_length = df.groupby(["task", "domain"])["curr_plan_length"].mean().reset_index()

        mean_plan_length.to_csv(f"{tables_dir}/mean_plan_length_by_task_domain.csv", index=False)
        self.debug_invlid_improved_plan(df)
        # self.debug_score_normalisation(df)
        # self.check_for_better_nlp_manipulation(df)

    def run_plots(self, df):
        """
        Run all the plots on this dataframe
        :param df: the dataframe to use
        :return:
        """
        ## TABLES
        self.validity_steps_across_tasks_latex_table(df) # DONE TABLE
        self.core_metrics_across_tasks_latex_table(df) # DONE TABLE
        self.core_metrics_across_tasks_latex_table(df, group_by="model") # DONE TABLE

        # LAST EXECUTABLE ACTION ANALYSIS
        self.last_exec_frequency_relative_to_length_high_and_low("curr_last_act", "curr_plan", 10) # DONE PDF

        # PLAN TRACE ACTION QUALITY ANALYSIS # DONE
        self.plot_percentage_distribution_with_lines_high_and_low() # DONE PDF
        self.plot_percentage_distribution_with_lines_high_and_low(plan_column="nr_plan_trace") # DONE PDF
        #
        # self.complemenary_length_out_of_gt_length(df) # DONE PDF
        self.complemenary_length_out_of_gt_length_high_and_low() # DONE PDF

        #
        #### APPENDIX ######
        self.plot_plan_length_distributions_by_domain(df) # DONE APPENDIX PDF

        # BIG TABLE
        if "o4-mini" in args.domain or args.domain == "":
            self.latex_one_summary_table_per_experiment_by_domain(df) # 1 TABLE FOR ALL RESULTS (APPENDIX)
        else:
            self.latex_one_summary_table_per_experiment(df) # 1 TABLE FOR ALL RESULTS (APPENDIX)

        # EXECUTABILITY ANALYSIS
        self.plot_executability_validity_stacked(df, "curr_validity", "executability", "curr_plan") # DONE APPENDIX
        self.plot_success_vs_last_executable(df)

        # O4 mini
        self.get_performance_by_problem_difficulty(df, "blocksworld")
        self.get_performance_by_problem_difficulty(df, "logistics")

        self.mathematical_analysis(df)

    def mathematical_analysis(self, results):
        """
        Function to perform some analysis computations on the results
        :param results: the DataFrame to use
        :return:
        """
        df = results.copy()

        # Get the number of plans where the first action is not executable
        df["last_exec_is_zero"] = df["curr_last_act"].apply(lambda x: 1 if x == 0 else 0)

        print("Number of plans where the first action is not executable:", df["last_exec_is_zero"].sum())
        print("Which is ", df["last_exec_is_zero"].sum() / len(df) * 100, "% of the plans")

        # Get the number of plans where the complementary plan is the same as the GT plan
        df["complementary_plan"] = df["complementary_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["complementary_plan"] = df.apply(lambda x: 1 if x["complementary_plan"] == x["gt_plan"] else 0, axis=1)
        print("Number of plans where the complementary plan is the same as the GT plan:", df["complementary_plan"].sum())
        print("Which is ", df["complementary_plan"].sum() / len(df) * 100, "% of the plans")

        # Get the number of plans where the correct part is empty
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_emptiness"] = df.apply(lambda x: 1 if len(x["correct_part"]) == 0 else 0, axis=1)
        print("Number of plans where the correct part is empty:", df["correct_part_emptiness"].sum())
        print("Which is ", df["correct_part_emptiness"].sum() / len(df) * 100, "% of the plans")

        # Get the overall success rate
        overall_success_rate = df["curr_validity"].mean()
        print(f"Overall success rate: {overall_success_rate * 100:.2f}% of the plans")
        # Improves success rate
        improved_success_rate = df["cdt_subseq_validity"].mean()
        print(f"Improved success rate after CDT Subsequent Plan: {improved_success_rate * 100:.2f}% of the plans")
        print(f"Improvement in success rate: {((improved_success_rate - overall_success_rate) / overall_success_rate) * 100:.2f}%")

        # Get average length of GT plans for each domain
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        avg_gt_plan_length = df.groupby("domain")["gt_plan"].apply(lambda x: np.mean([len(plan) for plan in x])).reset_index()
        print(f"Average length of GT plans for each domain:\n{avg_gt_plan_length}")
        # Get total average length of GT plans
        total_avg_gt_plan_length = df["gt_plan"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"Total length of GT plans for each domain:\n{total_avg_gt_plan_length}")

        # Get average length of correct part for each domain
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_length"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        avg_correct_part_length = df.groupby("domain")["correct_part_length"].mean().reset_index()
        print(f"Average length of correct part for each domain:\n{avg_correct_part_length}")
        # Get total average length of correct part
        total_avg_correct_part_length = df["correct_part_length"].mean()
        print(f"Total average length of correct part: {total_avg_correct_part_length}")

        # Get average LEA (Last Executable Action) for each domain
        df["curr_last_act"] = df["curr_last_act"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        avg_lea = df.groupby("domain")["curr_last_act"].mean().reset_index()
        print(f"Average Last Executable Action (LEA) for each domain:\n{avg_lea}")

        # Get average LEA for cdt_subseq for each domain
        df["cdt_subseq_last_act"] = df["cdt_subseq_last_act"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        avg_cdt_subseq_lea = df.groupby("domain")["cdt_subseq_last_act"].mean().reset_index()
        print(f"Average Last Executable Action (LEA) for CDT Subsequent Plan for each domain:\n{avg_cdt_subseq_lea}")
        # And in total
        total_avg_cdt_subseq_lea = df["cdt_subseq_last_act"].mean()
        print(f"Total average Last Executable Action (LEA) for CDT Subsequent Plan: {total_avg_cdt_subseq_lea}")

    def get_non_matching(self, gt_plan, improved_plan):
        return [a for a, b in zip(gt_plan, improved_plan) if a != b]

    def split_results_by_success_rate(self, success_column, group_by="model") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the results by success rate (validity) into 2 separate dataframes
        :param success_column: the column to use for splitting (curr_validity or another)
        :param group_by: the column to group by (default is "model")
        :return: 2 dataframes, one with higher success rates and one with lower success rates
        """

        df = deepcopy(self.results)
        success_rates = df.groupby(group_by)[success_column].mean().reset_index()

        threshold = success_rates[success_column].quantile(0.5) # get the nid threshold

        higher_rate = success_rates[success_rates[success_column] > threshold]

        higher_success = df.merge(higher_rate[group_by], on=[group_by], how="inner")
        lower_success = df[~df[[group_by]].apply(tuple, axis=1).isin(higher_rate[[group_by]].apply(tuple, axis=1))]

        high_results_file = self.results_file.replace(".csv", "_high_success.csv")
        low_results_file = self.results_file.replace(".csv", "_low_success.csv")

        if not os.path.exists(high_results_file):
            higher_success.to_csv(high_results_file, index=False)
        if not os.path.exists(low_results_file):
            lower_success.to_csv(low_results_file, index=False)

        return higher_success, lower_success

    """
    Success rate analysis
    """

    # Visualisation functions
    def validity_steps_across_tasks_latex_table(self, results):
        """
        Generate a LaTeX table of success rates and average steps per task for all plan types throughout recovery.
        :param results:
        :return:
        """
        df = results.copy()

        for col in self.success_columns:
            df[col] = df[col].astype(bool)

        step_columns = [col.replace("validity", "steps") for col in self.success_columns]
        plan_labels = [f"$\\pi_{i}$" for i in range(len(self.success_columns))]

        rows = []
        for task, group in df.groupby("task"):
            row = {"Task": task}
            for v_col, s_col, label in zip(self.success_columns, step_columns, plan_labels):
                row[f"{label} Validity"] = round(group[v_col].mean() * 100, 1)
                row[f"{label} Steps"] = int(round(group[s_col].mean()))
            rows.append(row)

        table_df = pd.DataFrame(rows)
        latex = table_df.to_latex(index=False, escape=False, column_format="l" + "cc" * len(self.success_columns),
                                  float_format = "%.2f")

        with open(f"{tables_dir}/compact_validity_steps_table.tex", "w") as f:
            f.write(latex)

        return latex

    def core_metrics_across_tasks_latex_table(self, results, group_by="task"):
        df = results.copy()

        # Safe eval for stringified lists
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        if group_by == "task":
            df = self.map_config_in_results(df, group_by="task")
        else: # model
            df = self.map_config_in_results(df, group_by="model")

        metrics = [
            "curr_validity", "curr_score", "trace_score", "curr_steps", "curr_last_act",
            "cdt_validity", "subseq_validity", "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act", "improved_validity",
        ]
        col_map = self.get_column_names_map()

        # Convert columns to numeric
        if group_by == "task":
            df = df[list(metrics) + ["task_name"]].copy()
            df = df.dropna()
            grouped = df.groupby("task_name").mean().round(2)
        else:
            df = df[list(metrics) + ["model_name"]].copy()
            df = df.dropna()
            grouped = df.groupby("model_name").mean().round(2)

        # Bold best values (max, except for steps to validity: min)
        def bold_best(series, minimise=False):
            if minimise:
                best = series.min()
            else:
                best = series.max()
            return series.apply(lambda x: f"\\textbf{{{x}}}" if x == best else f"{x}")

        formatted = grouped.copy()
        for metric in metrics:
            minimise = metric == "curr_steps" or metric == "cdt_subseq_steps"
            formatted[col_map[metric]] = bold_best(grouped[metric], minimise)

        formatted = formatted[[col_map[metric] for metric in metrics]]
        formatted = formatted.reset_index()

        latex = formatted.to_latex(
            index=False,
            escape=False,
            column_format="l" + "c" * (len(formatted.columns) - 1),
            caption=f"Comparison of core metrics by {group_by}",
            label=f"tab:core_metrics_by_{group_by}",
        )

        with open(f"{tables_dir}/compact_metrics_core_table_by_{group_by}.tex", "w") as f:
            f.write(latex)

        return latex

    def get_performance_by_problem_difficulty(self, results, domain):
        domain_replacement = domain # blocksworld_3
        domain_name = domain.replace("random_", "").replace("mystery_", "").replace("o4-mini", domain_replacement)
        domain_name = "blocksworld_3" if "blocksworld" in domain_name else domain_name
        goals_csv = os.path.join(results_dir, os.pardir, f"{domain_name}_goal_counts.csv") if domain_name else f"{results_dir}/../goal_counts.csv"
        goals_df = pd.read_csv(goals_csv, header=0)

        # self.plot_metric_by_group(results, goals_df, group_by="num_goals", domain=domain_name, metric="stv")
        self.plot_metric_by_group(results, goals_df, group_by="num_goals", domain=domain_name)
        # self.plot_metric_by_group(results, goals_df, group_by="num_goals", metric="LEA")
        # self.plot_metric_by_group(results, goals_df, group_by="gt_length", domain=domain_name, metric="stv")
        self.plot_metric_by_group(results, goals_df, group_by="gt_length", domain=domain_name)
        # self.plot_metric_by_group(results, goals_df, group_by="gt_length", metric="LEA")

    def plot_metric_by_group(self, results, goals_df, group_by="num_goals", domain="", metric="general"):
        """
        Plots multiple metrics (SR, LEA, StV, Score, AQM) grouped by goal count or ground truth length.

        :param results: DataFrame containing the results.
        :param goals_df: DataFrame with columns ["domain", "instance_id", "num_goals"]
        :param group_by: "num_goals" or "gt_length"
        """
        results_df = results.copy()
        results_df["instance_id"] = results_df["instance_id"].astype(int)
        goals_df["instance_id"] = goals_df["instance_id"].astype(int)

        # Normalize domain names if needed
        if "domain" in results_df.columns:
            results_df["domain"] = results_df["domain"].apply(
                lambda x: "blocksworld_3" if "blocksworld" in x else x
            )
            results_df["domain"] = results_df["domain"].apply(
                lambda x: "logistics" if "logistics" in x else x
            )

        df = pd.merge(results_df, goals_df, on=["domain", "instance_id"], how="inner")

        df["gt_length"] = df["gt_plan"].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x))
        df = df.dropna(subset=[group_by])

        for col in ["curr_validity", "curr_last_act", "curr_steps", "curr_score", "trace_score", "gt_plan"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        df = df.dropna(subset=["curr_validity", "curr_last_act", "curr_steps", "curr_score", "trace_score"])

        # df["LEA"] = df["curr_last_act"] / df["gt_length"]
        # invalid_df = df[df["curr_validity"] == False]
        # lea_grouped = invalid_df.groupby(group_by)["LEA"].mean().reset_index(name="LEA/GT")

        grouped = df.groupby(group_by).agg({
            "curr_validity": "mean",
            # "LEA": "mean",
            "curr_score": "mean",
            "curr_steps": "mean",
            # "trace_score": "mean",
            "instance_id": "count"
        }).rename(columns={
            "curr_validity": "SR",
            # "LEA": "LEA/GT",
            "curr_score": "Score",
            "curr_steps": "StV",
            # "trace_score": "AQM",
            "instance_id": "Count"
        }).reset_index()

        # grouped = pd.merge(grouped, lea_grouped, on=group_by, how="left")

        # Normalise each column (except the group_by and count)
        metrics_to_normalise = [col for col in grouped.columns if col not in [group_by, "instance_id"]]
        for col in metrics_to_normalise:
            min_val = grouped[col].min()
            max_val = grouped[col].max()
            if max_val > min_val:  # avoid division by zero
                grouped[col] = (grouped[col] - min_val) / (max_val - min_val)
            else:
                grouped[col] = 0.0  # or 1.0, depending on how you want to treat constant values

        # Plotting
        plt.figure(figsize=(10, 5))
        metrics = ["SR", "Score", "StV"] # , "LEA/GT" , "AQM"
        colors = ["skyblue", "green", "orange"] # , "salmon", "purple"

        for metric, color in zip(metrics, colors):
            plt.plot(grouped[group_by], grouped[metric], label=metric, marker="o", color=color)

        xlabel = "Number of Goals" if group_by == "num_goals" else "GT Plan Length"
        plt.xlabel(xlabel, fontsize=26)
        plt.ylabel("Mean Metric Value", fontsize=26)
        # plt.title(f"Evaluation Metrics vs. {xlabel}", fontsize=16)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/all_metrics_by_{group_by}_{domain}.pdf", format="pdf")
        plt.show()

    def extract_success_rates_at_lengths(self, results, plan_column, validity_column_1, validity_column_2,
                                         lengths=[2, 4, 6, 8]):
        df = results.copy()
        df[plan_column] = df[plan_column].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df["plan_length"] = df[plan_column].apply(len)

        grouped = df.groupby("plan_length")[[validity_column_1, validity_column_2, "curr_score", "curr_steps", "curr_last_act"]].mean()

        print("Success Rates by Plan Length:")
        for length in lengths:
            if length in grouped.index:
                val1 = grouped.loc[length, validity_column_1]
                val2 = grouped.loc[length, validity_column_2]
                score = grouped.loc[length, "curr_score"]
                stv = grouped.loc[length, "curr_steps"]
                lea = grouped.loc[length, "curr_last_act"]
                print(f"Length {length}: {validity_column_1} = {val1:.2f}, {validity_column_2} = {val2:.2f}, "
                      f"Score = {score:.2f}, Steps = {stv:.2f}, LEA = {lea:.2f}")
            else:
                print(f"Length {length}: No data available")

    """
    Score distribution analysis
    """
    def map_values_to_scores(self, results, column_name="plan_trace"):
        """
        Map the strings in the plan trace to their scores
        :return: the values mapping, the column name and the dataframe
        """
        df = results.copy()

        values = {
            "correct": 4,
            "misplaced": 2,
            "same_act": 1,
            "diff_act": 0.5,
            "redundant": 0
        }

        # get python dicts from string in csv
        df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        return values, column_name, df

    def plot_percentage_distribution_with_lines_high_and_low(self, cap=-1, plan_column="plan_trace"):
        """
        Plot the distribution of values across indices for both high and low success models.
        Also computes Spearman correlation between index and label percentages.
        """
        sorted_indices_h, values_matrix_h, rho_h, p_h, possible_values, cap_used = (
            self.prepare_data_index_distribution(self.high_success, plan_column, cap))
        sorted_indices_l, values_matrix_l, rho_l, p_l, _, _ = self.prepare_data_index_distribution(self.low_success, plan_column, cap_used)

        plt.figure(figsize=(10, 5))
        colors = {
            "correct": "#1f77b4",
            "same_act": "#ff7f0e",
            "diff_act": "#2ca02c",
            "redundant": "#d62728",
            "misplaced": "#9467bd",
        }

        for value in possible_values:
            plt.plot(sorted_indices_h, values_matrix_h[value], linestyle='-', linewidth=2,
                     color=colors.get(value, "gray"))
            plt.plot(sorted_indices_l, values_matrix_l[value], linestyle='--', linewidth=2,
                     color=colors.get(value, "gray"))

        plt.xlabel("Index", fontsize=26)
        plt.ylabel("Percentage (%)", fontsize=26)
        plt.title(
            f"$\\rho_\\mathrm{{high}}={rho_h:.2f}, p={p_h:.4f}$ | $\\rho_\\mathrm{{low}}={rho_l:.2f}, p={p_l:.4f}$",
            fontsize=24)

        print(f"\nHigh Success Spearman: $\\rho={rho_h:.2f}, p={p_h:.4f}$, plan_column={plan_column}")
        print(f"Low Success Spearman: $\\rho={rho_l:.2f}, p={p_l:.4f}$")

        # Compact legend components
        legend_elements = [
                              Line2D([0], [0], color='black', linestyle='-', label='High Success'),
                              Line2D([0], [0], color='black', linestyle='--', label='Low Success')
                          ] + [
                              Line2D([0], [0], color=colors[v], linestyle='-', label=v.capitalize()) for v in
                              possible_values
                          ]

        plt.legend(
            handles=legend_elements,
            loc='best',
            fontsize=16,
            frameon=True,
            ncol=1,
            handlelength=2,
            handletextpad=0.5,
            borderpad=0.4,
            labelspacing=0.3
        )

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()

        filename = f"{figures_dir}/percentage_distribution_with_lines_{plan_column}_compare_{cap_used}_high_and_low.pdf"
        plt.savefig(filename, format="pdf", dpi="figure")
        plt.show()

    def prepare_data_index_distribution(self, df, plan_column, cap):
        values, column_name, df = self.map_values_to_scores(df.copy(), plan_column)

        if plan_column == "nr_plan_trace":
            possible_values = ["correct", "same_act", "diff_act", "redundant"]
        else:
            possible_values = ["correct", "same_act", "diff_act", "redundant", "misplaced"]

        all_indices_flat = []
        all_indices = []
        all_scores = []
        data = []

        label_to_score = {
            "correct": 4,
            "misplaced": 3,
            "same_act": 2,
            "diff_act": 1,
            "redundant": 0,
        }

        for dict_data in df[column_name]:
            if isinstance(dict_data, dict):
                for index, label in dict_data.items():
                    if plan_column == "nr_plan_trace" and label == "misplaced":
                        label = "correct"
                    score = label_to_score.get(label)
                    if score is not None:
                        all_indices.append(index)
                        all_scores.append(score)
                        data.append((index, score))

        rho, p = (spearmanr(all_indices, all_scores) if all_indices else (None, None))

        for dict_data in df[column_name]:
            if isinstance(dict_data, dict):
                all_indices_flat.extend(dict_data.keys())
        if cap == -1 and all_indices_flat:
            all_indices_flat = np.array(all_indices_flat).astype(int)
            cap_val = int(np.percentile(all_indices_flat, 95))
        else:
            cap_val = cap

        value_counts = {}
        index_occurrences = {}

        for dict_data in df[column_name]:
            if isinstance(dict_data, dict):
                for index, value in dict_data.items():
                    if index >= cap_val:
                        continue
                    if plan_column == "nr_plan_trace" and value == "misplaced":
                        value = "correct"
                    if value == "position":
                        value = "correct"
                    if index not in value_counts:
                        value_counts[index] = {v: 0 for v in possible_values}
                        index_occurrences[index] = 0
                    value_counts[index][value] += 1
                    index_occurrences[index] += 1

        percentage_distribution = {
            idx: {value: (value_counts[idx][value] / index_occurrences[idx]) * 100 for value in possible_values}
            for idx in value_counts
        }

        sorted_indices = sorted(percentage_distribution.keys())
        values_matrix = {
            value: [percentage_distribution[idx].get(value, 0) for idx in sorted_indices]
            for value in possible_values
        }

        return sorted_indices, values_matrix, rho, p, possible_values, cap_val

    """
    Plan recovery analysis
    """
    def last_exec_frequency_relative_to_length_high_and_low(self, last_exec_column, plan_column, group=5):
        """
        Plot the frequency of each last executable action as a percentage of the total plan length,
        grouped into percentage levels, comparing high and low success models.
        """
        value_counts_high = self.prepare_data(self.high_success, last_exec_column, plan_column, group)
        value_counts_low = self.prepare_data(self.low_success, last_exec_column, plan_column, group)

        plt.figure(figsize=(10, 5))
        x = np.arange(len(value_counts_high.index))
        width = 0.35

        plt.bar(x - width / 2, value_counts_high.values, color="blue", alpha=0.45, width=width, label="High Success")
        plt.bar(x + width / 2, value_counts_low.values, color="blue", alpha=0.8, width=width, label="Low Success")

        plt.xlabel("Portion of executable part from full $\\pi_0$", fontsize=26)
        plt.ylabel("Frequency (%)", fontsize=26)
        plt.xticks(x, value_counts_high.index, rotation=45, ha="right", fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(fontsize=18)

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/last_exec_frequency_relative_to_length_h_and_l.pdf", dpi="figure", format="pdf")
        plt.show()

    def prepare_data(self, df, last_exec_column, plan_column, group=5):
        df = df.copy()
        df[plan_column] = df[plan_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        for col in [plan_column, "gt_plan"]:
            if col in df.columns:
                df = df[df[col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

        def compute_relative_position(row):
            if len(row[plan_column]) == 0:
                return 0
            return ((row[last_exec_column]) / len(row[plan_column])) * 100

        df["relative_position"] = df.apply(compute_relative_position, axis=1)

        bins = list(range(0, 101, group))
        labels = [f"{i}-{i + group}%" for i in bins[:-1]]

        df["grouped_position"] = pd.cut(df["relative_position"], bins=bins, labels=labels, include_lowest=True)
        value_counts = df["grouped_position"].value_counts(normalize=True) * 100
        value_counts = value_counts.sort_index()
        return value_counts

    def complemenary_length_out_of_gt_length_high_and_low(self):
        """
        Plots the mean proportion of complementary_plan length relative to gt_plan length as a smooth line
        with variance shading, grouped by last executable action.
        It includes both cases: with and without last executable action.
        """

        def prepare_df(df):
            df = df.copy()
            df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df["complementary_plan"] = df["complementary_plan"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df["improved_plan"] = df["improved_plan"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df["correct_part"] = df["correct_part"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            df["gt_length"] = df["gt_plan"].apply(len)
            df["complementary_length"] = df["complementary_plan"].apply(len)
            df["improved_length"] = df["improved_plan"].apply(len)
            df["correct_part_length"] = df["correct_part"].apply(len)

            df = df[(df["complementary_length"] > 0) & (df["correct_part_length"] > 0)]

            df["length_ratio_no_exec"] = df["complementary_length"] / df["gt_length"]
            df["length_ratio_correct_part"] = df["correct_part_length"] / df["gt_length"]

            cap_value = int(np.percentile(df["curr_last_act"], 99))
            df = df[df["curr_last_act"] <= cap_value]

            grouped_no_exec = df.groupby("curr_last_act")["length_ratio_no_exec"].agg(
                ["mean", "std", "count"]).reset_index()
            grouped_no_exec = grouped_no_exec[grouped_no_exec["count"] >= 3]

            grouped_correct = df.groupby("curr_last_act")["length_ratio_correct_part"].agg(
                ["mean", "std", "count"]).reset_index()
            grouped_correct = grouped_correct[grouped_correct["count"] >= 3]

            return grouped_no_exec, grouped_correct

        high_no_exec, high_correct = prepare_df(self.high_success)
        low_no_exec, low_correct = prepare_df(self.low_success)

        plt.figure(figsize=(10, 5))

        plt.plot(high_no_exec["curr_last_act"], high_no_exec["mean"], color="royalblue", linestyle="-", marker="o",
                 label="High $\\pi_{\\text{comp}}$")
        plt.fill_between(high_no_exec["curr_last_act"],
                         high_no_exec["mean"] - high_no_exec["std"],
                         high_no_exec["mean"] + high_no_exec["std"],
                         color="royalblue", alpha=0.2)

        plt.plot(high_correct["curr_last_act"], high_correct["mean"], color="green", linestyle="-", marker="^",
                 label="High $\\pi_{\\text{corr}}$")
        plt.fill_between(high_correct["curr_last_act"],
                         high_correct["mean"] - high_correct["std"],
                         high_correct["mean"] + high_correct["std"],
                         color="green", alpha=0.2)

        plt.plot(low_no_exec["curr_last_act"], low_no_exec["mean"], color="royalblue", linestyle="--", marker="o",
                 label="Low $\\pi_{\\text{comp}}$")
        plt.fill_between(low_no_exec["curr_last_act"],
                         low_no_exec["mean"] - low_no_exec["std"],
                         low_no_exec["mean"] + low_no_exec["std"],
                         color="royalblue", alpha=0.4)

        plt.plot(low_correct["curr_last_act"], low_correct["mean"], color="green", linestyle="--", marker="^",
                 label="Low $\\pi_{\\text{corr}}$")
        plt.fill_between(low_correct["curr_last_act"],
                         low_correct["mean"] - low_correct["std"],
                         low_correct["mean"] + low_correct["std"],
                         color="green", alpha=0.4)

        plt.xlabel("Last Executable Action Index", fontsize=26)
        plt.ylabel("Portion from $\\pi_{\\text{GT}}$", fontsize=26)
        plt.ylim(0, 2)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18, loc="upper right", frameon=False, ncol=2)
        plt.tight_layout()

        plt.savefig(f"{figures_dir}/complementary_length_out_of_gt_length_high_and_low.pdf", format="pdf",
                    dpi="figure")
        print(f"Saved complementary length plot for high and low success models to {figures_dir}/complementary_length_out_of_gt_length_high_and_low.pdf")
        plt.show()

    def plot_executability_validity_stacked(self, results, validity_column, executability_column, plan_column,
                                            group_by_column="model"):
        """
        Plot executability and validity per model/task with std and Spearman correlation across instances.

        :param results: The dataframe with results.
        :param validity_column: Column for plan validity (bool).
        :param executability_column: Column for executability (bool).
        :param plan_column: Column for full plan (stringified list).
        :param group_by_column: "model" or "task".
        """
        if group_by_column not in ["model", "task"]:
            raise ValueError("group_by_column must be either 'model' or 'task'")

        df = results.copy()

        if isinstance(df[plan_column].iloc[0], str):
            df[plan_column] = df[plan_column].apply(ast.literal_eval)

        df.loc[df[plan_column].apply(len) == 0, executability_column] = False

        # Spearman correlation per-instance
        spearman_corr, pval = spearmanr(df[executability_column], df[validity_column])
        print(f"Spearman correlation between {executability_column} and {validity_column}: "
                f"rho = {spearman_corr:.2f}, p = {pval:.3f}")

        df = self.abbreviate_config_names(df)
        instance_df = df.groupby(["instance_id", f"{group_by_column}_name"]).agg({
            executability_column: "mean",
            validity_column: "mean"
        }).reset_index()

        # Group by model/task
        grouped = instance_df.groupby(f"{group_by_column}_name")
        plot_df = grouped.agg(
            Executability=(executability_column, 'mean'),
            Validity=(validity_column, 'mean'),
            Executability_std=(executability_column, 'std'),
            Validity_std=(validity_column, 'std')
        ).reset_index()

        for col in ["Executability", "Validity", "Executability_std", "Validity_std"]:
            plot_df[col] *= 100

        plot_df = plot_df.sort_values("Executability", ascending=False).reset_index(drop=True)

        # Clamp error bars to [0, 100]
        lower_exec = np.clip(plot_df["Executability"] - plot_df["Executability_std"], 0, 100)
        upper_exec = np.clip(plot_df["Executability"] + plot_df["Executability_std"], 0, 100)
        exec_errors = [plot_df["Executability"] - lower_exec, upper_exec - plot_df["Executability"]]

        lower_valid = np.clip(plot_df["Validity"] - plot_df["Validity_std"], 0, 100)
        upper_valid = np.clip(plot_df["Validity"] + plot_df["Validity_std"], 0, 100)
        valid_errors = [plot_df["Validity"] - lower_valid, upper_valid - plot_df["Validity"]]

        fig, ax = plt.subplots(figsize=(10, 5) if group_by_column == "model" else (8, 6))
        bar_width = 0.6 if group_by_column == "model" else 0.5
        x = np.arange(len(plot_df))

        ax.bar(x - bar_width / 4, plot_df["Executability"], yerr=exec_errors, capsize=6,
               color="blue", alpha=0.7, width=bar_width / 2, label="Executable Plans (%)")

        ax.bar(x + bar_width / 4, plot_df["Validity"], yerr=valid_errors, capsize=6,
               color="crimson", alpha=0.8, width=bar_width / 2, label="Valid Plans (%)")

        ax.set_xlabel(group_by_column.capitalize(), fontsize=20)
        ax.set_ylabel("Percentage (%)", fontsize=20)
        # ax.set_title(f"Executability vs Validity per {group_by_column.capitalize()}\n" ... below
        ax.set_title(f"Spearman r = {spearman_corr:.2f} (p = {pval:.3f})", fontsize=20)

        # model_names = self.map_model_names(plot_df[group_by_column].tolist())
        model_names = plot_df[f"{group_by_column}_name"]
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="center", fontsize=13)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/executability_validity_stacked.pdf", format="pdf", dpi="figure")
        # plt.show()

    def plot_success_vs_last_executable(self, results):
        """
        Plots SR vs mean LEA for invalid plans with Spearman correlation,
        shown side-by-side for low and high success models.
        """

        def prepare_data(df):
            df = df.copy()
            df["curr_plan"] = df["curr_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df = df[df["curr_plan"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

            success_df = df.groupby(["domain", "model", "task"]).agg(
                success_rate=("curr_validity", "mean"),
                count=("curr_validity", "count")
            ).reset_index()

            invalid_only = df[df["curr_validity"] == False]
            last_exec_df = invalid_only.groupby(["domain", "model", "task"]).agg(
                mean_last_exec=("curr_last_act", "mean")
            )

            grouped = pd.merge(success_df, last_exec_df, on=["domain", "model", "task"], how="inner")
            # grouped = grouped[grouped["count"] >= 10]

            # grouped["model"] = grouped["model"].apply(self.map_model_names)
            return grouped

        def print_stats(grouped, label):
            print(f"=== LEA vs SR: {label.upper()} MODELS ===")
            quantiles = pd.qcut(grouped["success_rate"], q=3, duplicates="drop", retbins=True)
            num_bins = len(quantiles[1]) - 1

            # Define appropriate number of labels
            label_options = ["Low", "Medium", "High"]
            grouped["success_group"] = pd.qcut(grouped["success_rate"], q=3, labels=label_options[:num_bins], duplicates="drop")

            low = grouped[grouped["success_group"] == "Low"]["mean_last_exec"]
            med = grouped[grouped["success_group"] == "Medium"]["mean_last_exec"]
            high = grouped[grouped["success_group"] == "High"]["mean_last_exec"]
            H, p_kw = kruskal(low, med, high)
            print(f"Kruskal-Wallis H = {H:.2f}, p = {p_kw:.4f}")

            grouped["lea_bin"] = pd.qcut(grouped["mean_last_exec"], q=3, labels=["Short", "Medium", "Long"])
            contingency = pd.crosstab(grouped["success_group"], grouped["lea_bin"])
            chi2, p_chi2, _, _ = chi2_contingency(contingency)
            print(f"Chi-Square χ² = {chi2:.2f}, p = {p_chi2:.4f}")

            rho, p_rho = spearmanr(grouped["success_rate"], grouped["mean_last_exec"])
            tau, p_tau = kendalltau(grouped["success_rate"], grouped["mean_last_exec"])
            print(f"Spearman ρ = {rho:.2f}, p = {p_rho:.4f}")
            print(f"Kendall τ = {tau:.2f}, p = {p_tau:.4f}")
            print("=" * 10)
            return rho, p_rho

        # Prepare data
        low_grouped = prepare_data(self.low_success)
        high_grouped = prepare_data(self.high_success)
        results_grouped = prepare_data(results.copy())

        rho_low, p_low = print_stats(low_grouped, "Low Success")
        rho_high, p_high = print_stats(high_grouped, "High Success")
        rho_total, p_total = print_stats(results_grouped, "Total Success")

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, grouped, title, rho, p in zip(
                axs,
                [low_grouped, high_grouped],
                ["Low Success Models", "High Success Models"],
                [rho_low, rho_high],
                [p_low, p_high],
        ):
            sns.regplot(
                data=grouped,
                x="success_rate",
                y="mean_last_exec",
                ax=ax,
                scatter_kws={"alpha": 0.7},
                line_kws={"color": "darkred"},
                ci=95
            )

            grouped = self.abbreviate_config_names(grouped.copy())

            for _, row in grouped.iterrows():
                x_jitter = np.random.uniform(-0.01, 0.01)
                y_jitter = np.random.uniform(-0.2, 0.2)
                label_str = f"{row["model_name"]}/{row["task_name"]}" if args.domain else f"{row["domain_name"]}/{row["model_name"]}/{row["task_name"]}"
                ax.text(row["success_rate"] + x_jitter, row["mean_last_exec"] + y_jitter, label_str, fontsize=8, alpha=0.8)

            ax.set_title(f"{title}; Spearman ρ = {rho:.2f}, p = {p:.4f}", fontsize=20)
            ax.set_xlabel("Success Rate", fontsize=20)
            ax.grid(True, linestyle="--", alpha=0.5)

        axs[0].set_ylabel("Mean LEA for Invalid Plans", fontsize=20)
        axs[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/success_vs_last_executable_high_and_low.pdf", format="pdf", dpi="figure")
        plt.show()

    def plot_plan_length_distributions_by_domain(self, results, generated_col="curr_plan", gt_col="gt_plan"):
        """
        Plot the distribution of plan lengths for LLM-generated and ground truth plans per domain.
        :param results: DataFrame containing the results.
        :param generated_col: Column name for LLM-generated plans.
        :param gt_col: Column name for ground truth plans.
        """
        df = results.copy()
        domain_col = "domain" # where domain name is specified
        for col in [generated_col, gt_col]:
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

        # Create structured dataframe
        data = []
        for _, row in df.iterrows():
            domain = row[domain_col]
            domain = self.map_domain_names(domain)
            gen_plan = row[generated_col]
            gt_plan = row[gt_col]

            if isinstance(gen_plan, list):
                data.append({"Length": len(gen_plan), "Type": "LLM-Generated", "Domain": domain})
            if isinstance(gt_plan, list):
                data.append({"Length": len(gt_plan), "Type": "Ground Truth", "Domain": domain})

        length_df = pd.DataFrame(data)

        # Cap lengths at 95th percentile
        cap = length_df["Length"].quantile(0.95)
        length_df = length_df[length_df["Length"] <= cap]

        # Color and style mapping
        palette = {
            ("Blocksworld", "Ground Truth"): "#1f77b4",
            ("Blocksworld", "LLM-Generated"): "#aec7e8",
            ("Logistics", "Ground Truth"): "#ff7f0e",
            ("Logistics", "LLM-Generated"): "#ffbb78",
            ("Mystery BW", "Ground Truth"): "#2ca02c",
            ("Mystery BW", "LLM-Generated"): "#98df8a",
            ("Random BW", "Ground Truth"): "#d62728",
            ("Random BW", "LLM-Generated"): "#ff9896",
            ("Random Logistics", "Ground Truth"): "#9467bd",
            ("Random Logistics", "LLM-Generated"): "#c5b0d5",
        }

        plt.figure(figsize=(9, 5))

        for (domain, plan_type), group in length_df.groupby(["Domain", "Type"]):
            label = f"{plan_type} ({domain})"
            color = palette[(domain, plan_type)]
            sns.kdeplot(
                data=group,
                x="Length",
                fill=True,
                alpha=0.7,
                linewidth=2,
                label=label,
                color=color
            )

        # plt.title("Plan Length Distributions by Domain and Plan Type", fontsize=20)
        plt.xlabel("Plan Length", fontsize=26)
        plt.ylabel("Density", fontsize=26)
        # plt.legend(title="Plan Type + Domain", fontsize=18, title_fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/plan_length_distributions_by_domain.pdf", format="pdf", dpi="figure")
        plt.show()

    def map_domain_names(self, domain):
        """
        Maps domain names to their full names for better readability in plots.
        :param domain_list: List of domain names.
        :return: List of mapped domain names.
        """
        domain_map = {
            "blocksworld_3": "Blocksworld",
            "mystery_blocksworld_3": "Mystery BW",
            "random_blocksworld_3": "Random BW",
            "logistics": "Logistics",
            "obfuscated_randomized_logistics": "Random Logistics",
        }
        return domain_map.get(domain, domain)

    #### TABLE ONLY
    def latex_one_summary_table_per_experiment(self, results):
        """
        Creates a LaTeX table for appendix, grouped by model × task.
        Improves formatting with bolded best values per model, model-task column split, and midrules.
        """
        df = results.copy()
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Map names for models and tasks
        df = self.map_config_in_results(df, group_by="model")
        df = self.map_config_in_results(df, group_by="task")

        df["curr_validity"] = df["curr_validity"].astype(float)
        df["cdt_validity"] = df["cdt_validity"].astype(float)
        df["subseq_validity"] = df["subseq_validity"].astype(float)
        df["cdt_subseq_validity"] = df["cdt_subseq_validity"].astype(float)
        df["improved_validity"] = df["improved_validity"].astype(float)

        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        # Metrics and column names
        metrics = [
            "curr_validity", "curr_score", "trace_score", "curr_steps", "correct_part_len", "curr_last_act",
            "cdt_validity", "cdt_last_act", "subseq_validity", "cdt_subseq_validity", "cdt_subseq_steps", "improved_validity"
        ]
        col_map = self.get_column_names_map()

        latex_table = self.format_latex_table(
            df, metrics, col_map,
            "Mean values of evaluation metrics per experiment (model × task). "
            "Bold highlights the best value per model.",
            f"tab:appendix_metrics_per_experiment",
            f"{tables_dir}/appendix_metrics_per_experiment.tex"
        )

    def latex_one_summary_table_per_experiment_by_domain(self, results):
        """
        Creates a LaTeX table for appendix, grouped by domain x model × task.
        Improves formatting with bolded best values per model, model-task column split, and midrules.
        """
        df = results.copy()
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Map names for models and tasks
        df = self.map_config_in_results(df, group_by="model")
        df = self.map_config_in_results(df, group_by="task")
        df = self.map_config_in_results(df, group_by="domain")

        df["curr_validity"] = df["curr_validity"].astype(float)
        df["cdt_validity"] = df["cdt_validity"].astype(float)
        df["subseq_validity"] = df["subseq_validity"].astype(float)
        df["cdt_subseq_validity"] = df["cdt_subseq_validity"].astype(float)
        df["improved_validity"] = df["improved_validity"].astype(float)

        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        # Metrics and column names
        metrics = [
            "curr_validity", "curr_score", "trace_score", "curr_steps", "correct_part_len", "curr_last_act",
            "cdt_validity", "cdt_last_act", "subseq_validity", "cdt_subseq_validity", "cdt_subseq_steps", "improved_validity"
        ]
        col_map = self.get_column_names_map()

        latex_table = self.format_latex_table(
            df, metrics, col_map,
            "Mean values of evaluation metrics per experiment (domain x model × task). "
            "Bold highlights the best value per model.",
            f"tab:appendix_metrics_per_experiment_by_domain",
            f"{tables_dir}/appendix_metrics_per_experiment_by_domain.tex",
            group_by=["domain_name", "model_name", "task_name"]
        )

    def get_column_names_map(self):
        return {
            "curr_validity": "$\\pi_0$ SR $\\uparrow$",
            "curr_score": "Score $\\uparrow$",
            "trace_score": "AQM $\\uparrow$",
            "curr_steps": "StV $\\downarrow$",
            "correct_part_len": "len($\\pi_{\\text{corr}}$ $\\uparrow$",
            "curr_last_act": "LEA $\\uparrow$",
            "cdt_validity": "$\\pi_1$ SR $\\uparrow$",
            "cdt_last_act": "$\\pi_2$ LEA $\\uparrow$",
            "subseq_validity": "$\\pi_2$ SR $\\uparrow$",
            "cdt_subseq_validity": "$\\pi_3$ SR $\\uparrow$",
            "cdt_subseq_steps": "$\\pi_3$ StV $\\downarrow$",
            "cdt_subseq_last_act": "$\\pi_3$ LEA",
            "improved_validity": "$\\pi_4$ SR $\\uparrow$",
        }

    def abbreviate_config_names(self, df):
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
        }

        task_map = {
            "_one_shot": "1S",
            "_state_tracking": "CoT",
            "_pddl": "PDDL",
            "_zero_shot": "0S",
        }

        domain_map = {
            "blocksworld_3": "BW",
            "mystery_blocksworld_3": "MBW",
            "random_blocksworld_3": "RBW",
            "logistics": "Log",
        }

        df["task_name"] = df["task"].map(task_map).fillna(df["task"])
        df["model_name"] = df["model"].map(model_map).fillna(df["model"])
        df["domain_name"] = df["domain"].map(domain_map).fillna(df["domain"])

        return df

    def map_config_in_results(self, df, group_by="model"):
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
            # "dsqwen": "DeepSeek Qwen",
        }

        task_map = {
            "_one_shot": "One-Shot",
            "_state_tracking": "State Tracking",
            "_pddl": "PDDL",
            "_zero_shot": "Zero-Shot",
        }

        domain_map = {
            "blocksworld_3": "Blocksworld",
            "mystery_blocksworld_3": "Mystery BW",
            "random_blocksworld_3": "Random BW",
            "logistics": "Logistics",
            "obfuscated_randomized_logistics": "Random Logistics",
        }

        if group_by == "task":
            df["task_name"] = df["task"].map(task_map).fillna(df["task"])
        elif group_by == "domain":
            df["domain_name"] = df["domain"].map(domain_map).fillna(df["domain"])
        else: # model
            df["model_name"] = df["model"].map(model_map).fillna(df["model"])

        return df

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse plan recovery results")
    parser.add_argument("--missing_instances", action="store_true", default=False, help="Whether to include missing instances in the analysis")
    parser.add_argument("--domain", type=str, default="o4-mini", choices=["mystery" "random" "blocksworld", "logistics", "o4-mini", ""], help="Domain to filter results by (blocksworld or logistics)")
    parser.add_argument("--success_rate", type=str, default="", choices=["high_success", "low_success", ""],
                        help="Success level to filter results by (high_success, low_success, or all)")
    # parser.add_argument("--LRMs", alction="store_true", default=True, help="Whether to evaluate LLMs or LRMs (default: LLMs)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    missing_instances = args.missing_instances
    domain = args.domain  # "blocksworld" or "logistics" ("" for all domains)
    success_level = args.success_rate
    # LRMs = args.LRMs  # True for LRMs, False for LLMs

    current_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    # eval_dir = "_LRMs" if LRMs else ""
    results_dir = f"{project_dir}/results/plan_recovery/final_evaluation/"
    # results_dir = f"{project_dir}/results/plan_recovery/from_hpc/50_instances/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if domain:
        missing_instances_file = f"{results_dir}/full_evaluation_w_missing_instances_{domain}.csv"
    else:
        missing_instances_file = f"{results_dir}/full_evaluation_w_missing_instances.csv"

    # suffix = "_LRMs" if LRMs else ""
    suffix = f"_{success_level}" if success_level else ""

    if missing_instances:
        results_file = missing_instances_file
        if domain:
            figures_dir = f"{project_dir}/figures/plan_recovery/final_evaluation{suffix}/figures_w_missing_instances_{domain}"
            tables_dir = f"{project_dir}/tables/plan_recovery/final_evaluation{suffix}/tables_w_missing_instances_{domain}"
        else:
            figures_dir = f"{project_dir}/figures/plan_recovery/final_evaluation{suffix}/figures_w_missing_instances"
            tables_dir = f"{project_dir}/tables/plan_recovery/final_evaluation{suffix}/tables_w_missing_instances"
    else:
        if domain:
            results_file = f"{results_dir}/full_evaluation_{domain}.csv"
            figures_dir = f"{project_dir}/figures/plan_recovery/final_evaluation{suffix}/figures_{domain}"
            tables_dir = f"{project_dir}/tables/plan_recovery/final_evaluation{suffix}/tables_{domain}"
        else:
            results_file = f"{results_dir}/full_evaluation.csv"
            figures_dir = f"{project_dir}/figures/plan_recovery/final_evaluation{suffix}/figures_full"
            tables_dir = f"{project_dir}/tables/plan_recovery/final_evaluation{suffix}/tables_full"
        # results_file = f"{results_dir}/blocksworld_3_evaluation_gemini_2_flash_state_tracking.csv"

    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    # Combine all results into one file if it's not already
    if not os.path.exists(results_file):
        print("Results file does not yet exist, combining results")
        if missing_instances:
            print("WARNING: should create results with missing instances")
        else:
            combine_csv_results(results_file, results_dir, result_filter=domain)
    else:
        print(f"Results file already exists: {results_file}")

    results = pd.read_csv(results_file)
    if not missing_instances and not os.path.exists(missing_instances_file):
        fill_missing_instances(results, missing_instances_file)

    # results_dir = os.path.join(results_dir, os.pardir) if "o4-mini" in domain else results_dir
    analyser = AnalyseResults(results_file, results_dir)
    # analyser.save_row_to_text(f"{results_dir}/blocksworld_3_evaluation_qcode_one_shot.csv", 8) # QCODE one-shot instance 10
