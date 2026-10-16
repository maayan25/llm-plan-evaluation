import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import ast

from pandas import DataFrame
from scipy.stats import spearmanr, kruskal, chi2_contingency, kendalltau
from matplotlib.lines import Line2D

from utils import combine_csv_results, postprocess_results

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
    def __init__(self, results_file, results_dir, is_processed=False, success_level=None, domain=None, run_analysis=True, tables_dir=None, figures_dir=None):
        self.results_file = results_file
        self.results = pd.read_csv(results_file)
        self.domain = domain

        if not is_processed:
            self.global_results = pd.read_csv(f"{results_dir}/full_evaluation.csv")
            self.results = postprocess_results(self.results, self.global_results)

            # Save postprocessed results to a new CSV file
            postprocessed_file = results_file.replace(".csv", "_postprocessed.csv")
            self.results.to_csv(postprocessed_file, index=False)

        self.domain_name_mapping = {
            "mystery_bw": "blocksworld",
            "random_bw": "blocksworld",
            "deceptive_lg": "logistics",
            "random_lg": "logistics",
        }

        self.tables_dir = tables_dir
        self.figures_dir = figures_dir

        # Split the results by success rate
        self.success_columns = ["curr_validity", "cdt_validity", "subseq_validity", "cdt_subseq_validity", "improved_validity"]
        # if "o4-mini" in args.domain:
        #     self.high_success, self.low_success = self.split_results_by_success_rate("curr_validity", group_by="domain")
        # else:
        self.high_success, self.low_success = self.split_results_by_success_rate("curr_validity")

        # Run all the plots
        if success_level:
            self.results = self.high_success if success_level == "high_success" else self.low_success

        discard_empty_plans = True
        if discard_empty_plans:
            print(f"Discarding empty plans from the results. Original length: {len(self.results)}")
            self.results = self.results[self.results["curr_plan"].apply(lambda x: len(ast.literal_eval(x)) > 0 if isinstance(x, str) else len(x) > 0)]
            print(f"Length after discarding empty plans: {len(self.results)}")

        if run_analysis:
            # self.run_plots(self.results)
            self.get_analysis(self.results)

    def get_analysis(self, df):
        # Get the average success rates for each plan type
        success_means = {col: df[col].mean() for col in self.success_columns}

        # Get the average steps for each plan type
        step_columns = [col.replace("validity", "steps") for col in self.success_columns]
        step_means = {col: df[col].mean() for col in step_columns}

        # Save the analysis to a csv file
        plans = ["pi_LLM", "pi_NLP", "pi_LCS", "pi_NLP-LCS", "pi_VAL"]
        analysis_df = pd.DataFrame({
            "Plan Type": plans,
            "Success Rate": [f"{success_means[col]:.2%}" for col in self.success_columns],
            "Average Steps": [f"{step_means[col]:.2f}" for col in step_columns]
        })

        analysis_df.to_csv(f"{self.tables_dir}/analysis_results.csv", index=False)

        self.extract_success_rates_at_lengths(df, "curr_validity", "cdt_subseq_validity", plan_column="curr_plan", group_by="curr_plan")  # DONE TABLE for BW
        self.extract_success_rates_at_lengths(df, "curr_validity", "cdt_subseq_validity", group_by="optimal")  # DONE TABLE for BW

        # Mean plan length for Logistics CoT VS Logistics one-shot
        df["curr_plan"] = df["curr_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["curr_plan_length"] = df["curr_plan"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        mean_plan_length = df.groupby(["task", "domain"])["curr_plan_length"].mean().reset_index()

        mean_plan_length.to_csv(f"{self.tables_dir}/mean_plan_length_by_task_domain.csv", index=False)
        self.debug_invlid_improved_plan(df)
        # self.debug_score_normalisation(df)
        # self.check_for_better_nlp_manipulation(df)
        # self.find_examples_for_analysis(df) # Uncomment for files to find a running example of the analysis
        self.mathematical_analysis(df)

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
        self.core_metrics_across_tasks_latex_table(df, group_by="domain") # DONE TABLE

        self.core_metrics_across_tasks_latex_table_new(df) # DONE TABLE
        self.core_metrics_across_tasks_latex_table_new(df, group_by="model") # DONE TABLE
        self.core_metrics_across_tasks_latex_table_new(df, group_by="domain") # DONE TABLE

        if not self.domain == "baseline":
            self.compute_metric_correlations(df, group_by="task")
            self.plot_metric_correlation_heatmap(df, group_by="task")

            self.compute_metric_correlations(df, group_by="domain")
            self.plot_metric_correlation_heatmap(df, group_by="domain")

        self.compute_metric_correlations(df, group_by="model")
        self.plot_metric_correlation_heatmap(df, group_by="model")

        # self.bootstrap_correlation_difference(df, metric_a="curr_score")
        self.bootstrap_correlation_difference(df)
        # self.bootstrap_correlation_difference(df, metric_a="final_score_validity")
        # self.bootstrap_correlation_difference(df, metric_a="curr_score", metric_b="executability")
        self.bootstrap_correlation_difference(df, metric_b="executability")
        # self.bootstrap_correlation_difference(df, metric_a="final_score_validity", metric_b="executability")

        if self.domain == "baseline":
            self.last_exec_frequency_relative_to_length(df, "curr_last_act", "curr_plan", 10) # DONE PDF
            self.plot_percentage_distribution_with_lines(df) # DONE BW: Spearman rho=-0.43, p=0.0000 PDF
            self.plot_percentage_distribution_with_lines(df, plan_column="nr_plan_trace") # DONE BW: Spearman rho=-0.12, p=0.0000 PDF
            self.complemenary_length_out_of_gt_length(df) # DONE PDF
        else:
            self.last_exec_frequency_relative_to_length_high_and_low("curr_last_act", "curr_plan", 10)  # DONE PDF
            self.plot_percentage_distribution_with_lines_high_and_low() # DONE PDF
            self.plot_percentage_distribution_with_lines_high_and_low(plan_column="nr_plan_trace") # DONE PDF
            self.complemenary_length_out_of_gt_length_high_and_low() # DONE PDF

        #
        # # #### APPENDIX ######
        self.plot_plan_length_distributions_by_domain(df) # DONE APPENDIX PDF
        # self.success_rate_comparison_across_plan_types(df) # DONE APPENDIX PDF
        # BIG TABLE
        if not self.domain or self.domain == "baseline":
            self.latex_one_summary_table_per_experiment_by_domain(df) # 1 TABLE FOR ALL RESULTS (APPENDIX)
        else:
            self.latex_one_summary_table_per_experiment(df) # 1 TABLE FOR ALL RESULTS (APPENDIX)
            self.plot_success_vs_last_executable(df)
        #
        # EXECUTABILITY ANALYSIS
        self.plot_executability_validity_stacked(df, "curr_validity", "executability", "curr_plan") # DONE APPENDIX
        # stats_df = self.validate_claims(df) # per experiment NOT USED
        # self.plot_p_valid_given_exec_histogram(stats_df) # NOT USED
        # self.plot_valid_given_exec_vs_success() # NOT USED
        #

        domains = ["mystery_bw", "random_bw", "blocksworld_3", "blocksworld", "logistics", "elevator", "satellite", "deceptive_lg", "random_lg"]
        if not self.domain or self.domain == "baseline":
            print(f"Skipping performance by problem difficulty analysis for all domains. Please specify a domain to run this analysis.")
            # for domain in domains:
            #     self.get_performance_by_problem_difficulty(df, domain)
        else:
            self.get_performance_by_problem_difficulty(df, self.domain)

        self.mathematical_analysis(df)

    def mathematical_analysis(self, results):
        """
        Function to perform some analysis computations on the results
        :param results: the DataFrame to use
        :return:
        """
        df = results.copy()

        # Number of invalid plans
        num_invalid_plans = len(df[df["curr_validity"] == 0])
        print(f"Number of invalid plans: {num_invalid_plans} out of {len(df)} total plans ({num_invalid_plans / len(df) * 100:.2f}%)")
        # Number of executable plans
        num_exec = len(df[df["executability"] == 1])
        print(f"Number of executable plans: {num_exec} out of {len(df)} total plans ({num_exec / len(df) * 100:.2f}%)")

        # Number of plans that are invalid but executable
        num_invalid_exec = len(df[(df["curr_validity"] == 0) & (df["executability"] == 1)])
        print(f"Number of plans that are invalid but executable: {num_invalid_exec} out of {len(df)} total plans ({num_invalid_exec / len(df) * 100:.2f}%)")
        # Out of these, for how many is len(correct_part) > 0
        num_invalid_exec_correct_part = len(df[(df["curr_validity"] == 0) & (df["executability"] == 1) & (df["correct_part"].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x)) > 0)])
        print(f"Number of plans that are invalid but executable and have a non-empty correct part: {num_invalid_exec_correct_part} out of {len(df)} total plans ({num_invalid_exec_correct_part / len(df) * 100:.2f}%)")

        # Get the number of plans where the first action is not executable
        df["last_exec_is_zero"] = df["curr_last_act"].apply(lambda x: 1 if x == 0 else 0)

        print("Number of plans where the first action is not executable:", df["last_exec_is_zero"].sum())
        print("Which is ", df["last_exec_is_zero"].sum() / len(df) * 100, "% of the plans")
        # Get the number of plans where LEA < 0.5
        df["last_exec_less_than_half"] = df["curr_last_act"].apply(lambda x: 1 if x < 0.5 else 0)
        print(f"Number of plans where the Last Executable Action (LEA) is less than 0.5: {df['last_exec_less_than_half'].sum()}")
        print(f"Which is {df['last_exec_less_than_half'].sum() / len(df) * 100:.2f}% of the plans")

        # Get the number of plans where the complementary plan is the same as the GT plan
        df["complementary_plan"] = df["complementary_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["optimal_plan"] = df["optimal_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["complementary_plan"] = df.apply(lambda x: 1 if x["complementary_plan"] == x["optimal_plan"] else 0, axis=1)
        print("Number of plans where the complementary plan is the same as the optimal GT plan:", df["complementary_plan"].sum())
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

        per_metric_corr, mean_abs_corr = self.get_overall_metric_correlations(self.results)
        print(f"Overall metric correlations:\n{per_metric_corr}")
        # print(f"Mean absolute correlation across all metrics: {mean_abs_corr}") already in the method

        # Get average length of GT plans for each domain
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        avg_gt_plan_length = df.groupby("domain")["gt_plan"].apply(lambda x: np.mean([len(plan) for plan in x])).reset_index()
        print(f"Average length of GT plans for each domain:\n{avg_gt_plan_length}")
        # Get total average length of GT plans
        total_avg_gt_plan_length = df["gt_plan"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"Total length of GT plans for all domains:\n{total_avg_gt_plan_length}")

        # Get average length of optimal GT plans for each domain
        df["optimal_plan"] = df["optimal_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        avg_optimal_plan_length = df.groupby("domain")["optimal_plan"].apply(lambda x: np.mean([len(plan) for plan in x])).reset_index()
        print(f"Average length of optimal GT plans for each domain:\n{avg_optimal_plan_length}")
        # Get total average length of optimal GT plans
        total_avg_optimal_plan_length = df["optimal_plan"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"Total length of optimal GT plans for all domains:\n{total_avg_optimal_plan_length}")

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
        # And in total
        total_avg_lea = df["curr_last_act"].mean()
        if not "denormalised_curr_lea" in df.columns:
            print("denormalised_curr_lea column not found, calculating denormalised LEA using plan length")
            avg_plan_length = df["plan_length"].mean()
            denormalised_avg_lea = total_avg_lea * avg_plan_length
        else:
            denormalised_avg_lea = df["denormalised_curr_lea"].mean()
        print(f"Total average Last Executable Action (LEA): {total_avg_lea}, denormalised: {denormalised_avg_lea}")


        # Get average LEA for cdt_subseq for each domain
        df["cdt_subseq_last_act"] = df["cdt_subseq_last_act"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        avg_cdt_subseq_lea = df.groupby("domain")["cdt_subseq_last_act"].mean().reset_index()
        print(f"Average Last Executable Action (LEA) for CDT Subsequent Plan for each domain:\n{avg_cdt_subseq_lea}")
        # And in total
        total_avg_cdt_subseq_lea = df["cdt_subseq_last_act"].mean()
        print(f"Total average Last Executable Action (LEA) for CDT Subsequent Plan: {total_avg_cdt_subseq_lea}")

        # Number of plans where curr_steps > length of curr_plan
        df["curr_plan"] = df["curr_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        df["curr_steps"] = df["curr_steps"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["curr_steps_more_than_curr_plan"] = df.apply(lambda x: 1 if x["curr_steps"] > (len(x["curr_plan"]) + len(x["gt_plan"])) else 0, axis=1)
        print(f"Number of plans where curr_steps > (len(curr_plan) + len(gt_plan)): {df['curr_steps_more_than_curr_plan'].sum()} out of {len(df)} total plans ({df['curr_steps_more_than_curr_plan'].sum() / len(df) * 100:.2f}%)")

    def debug_invlid_improved_plan(self, results):
        """
        Debugging function to check the invalid improved plans
        :param df:
        :return:
        """
        df = results.copy()
        # Get all cases where improved plan is not the same as the GT plan and it is invalid
        df["improved_plan"] = df["improved_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        different_plans = df[df.apply(lambda x: x["improved_plan"] != x["gt_plan"], axis=1)]
        invalid_different_plans = different_plans[different_plans["improved_validity"] == 0]

        # Add a column for the diff between the improved plan and the GT plan
        # invalid_different_plans = self.add_diff_plan_column(invalid_different_plans)

        # Save the invalid different plans to a csv file
        invalid_different_plans = invalid_different_plans[["domain", "model", "task", "instance_id", "gt_plan", "improved_plan",
                                                           "improved_validity", "curr_plan", "curr_validity"]]
                                                           # "difference", "improved_validity", "curr_plan", "curr_validity"]]
        invalid_different_plans.to_csv(f"{self.tables_dir}/invalid_different_plans.csv", index=False)

        # # Save the different plans to a csv file
        # different_plans = different_plans[["instance_id", "gt_plan", "curr_plan", "improved_plan", "curr_validity", "improved_validity"]]
        # different_plans.to_csv(f"{self.tables_dir}/different_plans.csv", index=False)
        #
        # Save all cases where improved plan is invalid
        invalid_improved_plans = df[df["improved_validity"] == 0]
        invalid_improved_plans = invalid_improved_plans[["domain", "model", "task", "instance_id", "gt_plan", "curr_plan", "improved_plan", "curr_validity", "improved_validity"]]
        invalid_improved_plans.to_csv(f"{self.tables_dir}/invalid_improved_plans.csv", index=False)



    def find_examples_for_analysis(self, df):
        """
        Find rows with as many different values in the plan trace as possible
        :param df: the dataframe to use
        :return:
        """
        df["plan_trace"] = df["plan_trace"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Rows with all 5 values
        all_values_rows = df[df["plan_trace"].apply(lambda d: self.has_all_or_most_values(d, 5))]
        four_values_rows = df[df["plan_trace"].apply(lambda d: self.has_all_or_most_values(d, 4))]
        three_values_rows = df[df["plan_trace"].apply(lambda d: self.has_all_or_most_values(d, 3))]

        print("Rows with all 5 values:", len(all_values_rows))
        print("Rows with at least 4 values:", len(four_values_rows))

        suffix = "newest"
        all_values_rows.to_csv(f"all_values_rows_{suffix}.csv", index=False)
        four_values_rows.to_csv(f"four_values_rows_{suffix}.csv", index=False)
        three_values_rows.to_csv(f"three_values_rows_{suffix}.csv", index=False)

    def has_all_or_most_values(self, d, required_values=5):
        possible_values = {"misplaced", "correct", "same_act", "diff_act", "redundant"}
        if not isinstance(d, dict):  # Ensure it's a dictionary
            return False
        unique_values = set(d.values())  # Get the unique values in the dictionary
        return len(unique_values.intersection(possible_values)) >= required_values

    # Visualisation functions
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

    def success_rate_comparison_across_plan_types(self, results):
        """
        Compare the success rate across current plan, candidate plan, subseq plan, and subseq candidate plan,
        with standard deviation.
        :param results: the dataframe to use
        """
        df = results.copy()

        # Compute mean and standard deviation (without pi 4)
        success_means = {col: df[col].mean() for col in self.success_columns[:4]}
        success_stds = {col: df[col].std() for col in self.success_columns[:4]}

        # Define column labels
        latex_labels = [r"$\pi_{\text{LLM}}$", r"$\pi_{\text{NLP}}$", r"$\pi_{\text{LCS}}$", r"$\pi_{\text{NLP-LCS}}$"]
        columns = list(success_means.keys())
        means = [success_means[col] for col in columns]
        stds = [success_stds[col] for col in columns]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(latex_labels, means, yerr=stds, capsize=6,
                       color=['royalblue', 'forestgreen', 'indianred', 'purple'], alpha=0.8)

        # Formatting
        plt.xlabel("Plan Recovery Steps", fontsize=20)
        plt.ylabel("Average Success Rate", fontsize=20)
        # plt.title("Comparison of SR Throughout Plan Recovery", fontsize=20)
        plt.ylim(0, 1)
        plt.xticks(fontsize=20)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Place the value labels on top of the bars (not on top of the error bars)
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 3, height + 0.02,
                     f"{mean:.2f}", ha='right', fontsize=20)

        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/success_rate_comparison_across_plan_types.pdf", format="pdf", dpi="figure")
        # plt.show()

    def success_rate_comparison_across_plan_types_for_2_dfs(self, df1, df2, df1_label="hs", df2_label="ls"):
        """
        Compare the success rate across different plan recovery steps for two dataframes.
        Each success column will have two bars (one per df).

        :param df1: First dataframe to compare
        :param df2: Second dataframe to compare
        :param df1_label: Label for the first dataframe
        :param df2_label: Label for the second dataframe
        """
        df1_success_averages = {col: df1[col].mean() for col in self.success_columns}
        df2_success_averages = {col: df2[col].mean() for col in self.success_columns}

        labels = self.success_columns
        x = np.arange(len(labels)) # X positions for groups
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width / 2, df1_success_averages.values(), width, label=df1_label, color='royalblue', alpha=0.8)
        bars2 = ax.bar(x + width / 2, df2_success_averages.values(), width, label=df2_label, color='seagreen', alpha=0.8)
        ax.set_xlabel("Plan Recovery Steps")
        ax.set_ylabel("Average Success Rate")
        ax.set_title("Comparison of Success Rates Throughout Plan Recovery")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim(0, 1)  # Success rate between 0 and 1
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

        # Add text annotations
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.2f}", ha='center', fontsize=14)

        plt.savefig(f"{self.figures_dir}/success_rate_comparison_across_plan_types_for_2_dfs.png", dpi=300)
        plt.show()

    def success_rate_across_tasks_all_plans_table_latex(self, results):
        """
        Generate a LaTeX table of success rates per task for all plan types throughout recovery.
        :param results: The dataframe to use.
        :return: A LaTeX formatted table as a string.
        """
        df = results.copy()

        for col in self.success_columns:
            df[col] = df[col].astype(bool)

        df = df.dropna(subset=["task"])
        success_rates = df.groupby("task")[self.success_columns].mean().reset_index()

        # Convert success rates to percentages
        success_rates[self.success_columns] = success_rates[self.success_columns] * 100

        latex_table = success_rates.to_latex(index=False, float_format="%.2f",
                                             column_format="l" + "c" * len(self.success_columns))

        filename = f"{self.tables_dir}/success_rate_across_tasks_all_plans.tex"
        with open(filename, "w") as f:
            f.write(latex_table)
        return latex_table

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

        with open(f"{self.tables_dir}/compact_validity_steps_table.tex", "w") as f:
            f.write(latex)

        return latex

    def compute_metric_correlations(self, results: object, group_by: object = None, method: object = "pearson",
                                    primary_metrics: object = ("curr_validity", "executability", "final_score"),
                                    other_metrics: object = None, save_table: object = True) -> DataFrame:
        # "curr_score", "final_score_validity"
        df = results.copy()

        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        all_metrics = [
            "curr_validity", "executability", "final_score", "trace_score", "curr_steps", "curr_last_act",
            "correct_part_len", "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act", "improved_validity",
        ]
        if other_metrics is None:
            other_metrics = [m for m in all_metrics if m not in primary_metrics]
        needed = list(primary_metrics) + list(other_metrics)

        print(f"Computing correlations for metrics: {needed}")

        for col in needed:
            if df[col].dtype == object:
                df[col] = df[col].replace({"True": 1, "False": 0, True: 1, False: 0}).astype(float)
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)

        group_col_map = {"task": "task_name", "model": "model_name", "domain": "domain"}
        if group_by is not None:
            if group_by not in group_col_map:
                raise ValueError(f"Unsupported group_by '{group_by}'. Choose from {list(group_col_map)} or None.")
            group_col = group_col_map[group_by]
            df = df[needed + [group_col]].dropna()
            groups = sorted(df[group_col].unique())
        else:
            df = df[needed].dropna()
            groups = [None]
            group_col = None

        full_corr_matrices = {}
        summary_rows = {}

        for group in groups:
            subset = df if group is None else df[df[group_col] == group]
            if len(subset) < 2:
                continue
            corr_matrix = subset[needed].corr(method=method)
            key = group if group is not None else "Overall"
            full_corr_matrices[key] = corr_matrix

            row = {}
            for primary in primary_metrics:
                corr_values = corr_matrix.loc[primary, other_metrics].copy()
                # Flip sign only for *_steps metrics, which are inherently inversely
                # related to success by construction; leave all other signs as-is.
                for metric in other_metrics:
                    if metric.endswith("_steps"):
                        corr_values[metric] = -corr_values[metric]
                row[primary] = corr_values.mean().round(2)
            summary_rows[key] = row

        summary_df = pd.DataFrame(summary_rows).T.round(3)
        summary_df.index.name = group_by.capitalize() if group_by else "Group"

        self.last_full_corr_matrices = full_corr_matrices

        print(
            f"Mean correlations of {', '.join(primary_metrics)} with remaining metrics (steps metrics sign-flipped):\n{summary_df}")

        if save_table and self.tables_dir:
            col_map = self.get_column_names_map()

            def bold_best_row(row):
                best = row.max()
                return row.apply(lambda x: f"\\textbf{{{x:.2f}}}" if x == best else f"{x:.2f}")

            formatted = summary_df.apply(bold_best_row, axis=1)
            formatted = formatted.rename(columns={m: col_map.get(m, m) for m in primary_metrics})
            formatted = formatted.reset_index()

            latex = formatted.to_latex(
                index=False,
                escape=False,
                float_format="%.2f",
                column_format="l" + "c" * (len(formatted.columns) - 1),
                caption=f"Mean correlation of {', '.join(primary_metrics)} with remaining metrics (steps sign-flipped)"
                        + (f", by {group_by}" if group_by else ""),
                label="tab:metric_correlations" + (f"_by_{group_by}" if group_by else "_overall"),
            )
            suffix = f"_by_{group_by}" if group_by else "_overall"
            with open(f"{self.tables_dir}/metric_correlations{suffix}.tex", "w") as f:
                f.write(latex)

        return summary_df

    def plot_metric_correlation_heatmap(self, results, group_by="domain",
                                        primary_metrics=("curr_validity", "executability", "final_score"),
                                        other_metrics=None, method="pearson", save=True):
        """
        Heatmap: rows = group_by values (domain/task/model), columns = primary_metrics,
        values = mean correlation with the rest of the metrics (*_steps metrics
        sign-flipped before averaging, since they're inherently inversely related
        to success by construction).
        Self-contained: recomputes correlations each call, no shared state.
        """
        df = results.copy()

        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        all_metrics = [
            "curr_validity", "executability", "final_score", "trace_score", "curr_steps", "curr_last_act",
            "correct_part_len", "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act", "improved_validity",
        ]
        if other_metrics is None:
            other_metrics = [m for m in all_metrics if m not in primary_metrics]
        needed = list(primary_metrics) + list(other_metrics)

        for col in needed:
            if df[col].dtype == object:
                df[col] = df[col].replace({"True": 1, "False": 0, True: 1, False: 0}).astype(float)
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)

        group_col_map = {"task": "task_name", "model": "model_name", "domain": "domain"}
        if group_by not in group_col_map:
            raise ValueError(f"Unsupported group_by '{group_by}'. Choose from {list(group_col_map)}.")
        group_col = group_col_map[group_by]

        df = df[needed + [group_col]].dropna()

        rows = {}
        for group, subset in df.groupby(group_col):
            if len(subset) < 2:
                continue
            corr_matrix = subset[needed].corr(method=method)
            group_row = {}
            for primary in primary_metrics:
                corr_values = corr_matrix.loc[primary, other_metrics].copy()
                for metric in other_metrics:
                    if metric.endswith("_steps"):
                        corr_values[metric] = -corr_values[metric]
                group_row[primary] = corr_values.mean()
            rows[group] = group_row

        summary_df = pd.DataFrame(rows).T.round(2)
        summary_df = summary_df.sort_index()

        col_map = self.get_column_names_map()
        summary_df.columns = [col_map.get(m, m) for m in summary_df.columns]

        fig, ax = plt.subplots(figsize=(max(4, len(summary_df.columns) * 1.5), max(3, len(summary_df) * 0.5)))
        sns.heatmap(summary_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Mean correlation with other metrics (steps sign-flipped), by {group_by}")
        plt.tight_layout()
        if save and self.figures_dir:
            plt.savefig(f"{self.figures_dir}/metric_correlation_heatmap_by_{group_by}.pdf")
        plt.show()

        return summary_df

    def bootstrap_correlation_difference(self, results, metric_a="final_score", metric_b="curr_validity",
                                         other_metrics=None, n_boot=2000, method="pearson", seed=42):
        """
        Bootstraps the difference in mean correlation-with-other-metrics between
        two metrics, by resampling rows (instances) with replacement.
        Returns the observed difference, a 95% CI, and the fraction of bootstrap
        samples where metric_a > metric_b (an empirical p-value proxy).
        """
        rng = np.random.default_rng(seed)
        df = results.copy()

        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        all_metrics = [
            "curr_validity", "executability", "final_score", "trace_score", "curr_steps", "curr_last_act",
            "correct_part_len", "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act", "improved_validity",
        ]
        primary_metrics = (metric_a, metric_b)
        if other_metrics is None:
            other_metrics = [m for m in all_metrics if m not in primary_metrics]
        needed = list(primary_metrics) + list(other_metrics)

        for col in needed:
            if df[col].dtype == object:
                df[col] = df[col].replace({"True": 1, "False": 0, True: 1, False: 0}).astype(float)
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)

        df = df[needed].dropna().reset_index(drop=True)

        def mean_corr(subset, metric):
            corr_values = subset[needed].corr(method=method).loc[metric, other_metrics].copy()
            for m in other_metrics:
                if m.endswith("_steps"):
                    corr_values[m] = -corr_values[m]
            return corr_values.mean()

        observed_diff = mean_corr(df, metric_a) - mean_corr(df, metric_b)

        boot_diffs = np.empty(n_boot)
        n = len(df)
        for i in range(n_boot):
            sample = df.iloc[rng.integers(0, n, n)]
            boot_diffs[i] = mean_corr(sample, metric_a) - mean_corr(sample, metric_b)

        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        p_proxy = (boot_diffs <= 0).mean()  # fraction of bootstrap draws where metric_a doesn't beat metric_b

        print(f"Observed difference ({metric_a} - {metric_b}): {observed_diff:.3f}")
        print(f"95% bootstrap CI: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"Fraction of bootstrap samples where difference <= 0: {p_proxy:.4f}")

        return observed_diff, (ci_low, ci_high), p_proxy

    def get_overall_metric_correlations(self, results,
                                        primary_metrics=("curr_validity", "executability", "final_score"),
                                        other_metrics=None, method="pearson"):
        """
        Computes correlation of each primary metric with every other metric, individually, over the whole (ungrouped) dataset.

        Returns:
            per_metric_corr: DataFrame, rows = primary_metrics, columns = other_metrics
            mean_corr: Series, mean correlation per primary metric (headline number)
        """
        df = results.copy()

        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        all_metrics = [
            "curr_validity", "executability", "final_score", "trace_score", "curr_steps", "curr_last_act",
            "correct_part_len", "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act", "improved_validity",
        ]
        if other_metrics is None:
            other_metrics = [m for m in all_metrics if m not in primary_metrics]
        needed = list(primary_metrics) + list(other_metrics)

        for col in needed:
            if df[col].dtype == object:
                df[col] = df[col].replace({"True": 1, "False": 0, True: 1, False: 0}).astype(float)
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)

        df = df[needed].dropna()

        corr_matrix = df.corr(method=method)
        per_metric_corr = corr_matrix.loc[list(primary_metrics), other_metrics].copy()

        for metric in other_metrics:
            if metric.endswith("_steps"):
                per_metric_corr[metric] = -per_metric_corr[metric]

        per_metric_corr = per_metric_corr.round(3)
        mean_corr = per_metric_corr.mean(axis=1).round(3)

        print(f"Mean correlation with other metrics ({method}, steps sign-flipped):")
        for metric, val in mean_corr.items():
            print(f"  {metric}: {val}")

        return per_metric_corr, mean_corr

    def core_metrics_across_tasks_latex_table_new(self, results, group_by="task"):
        df = results.copy()

        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        metrics = [
            "curr_validity", "executability", "final_score", "trace_score", "curr_steps", "curr_last_act", "correct_part_len",
            "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act", "improved_validity",
        ]
        col_map = self.get_column_names_map()

        for col in metrics:
            if df[col].dtype == object:
                print(col, df[col].unique()[:20])

        # Map the group_by option to the dataframe column
        group_col_map = {
            "task": "task_name",
            "model": "model_name",
            "domain": "domain",
        }
        if group_by not in group_col_map:
            raise ValueError(f"Unsupported group_by '{group_by}'. Choose from {list(group_col_map)}.")
        group_col = group_col_map[group_by]

        df = df[list(metrics) + [group_col]].copy()
        df = df.dropna()
        grouped = df.groupby(group_col).mean().round(2)

        # Bold best values (max, except for steps to validity: min)
        def bold_best(series, minimise=False):
            best = series.min() if minimise else series.max()
            return series.apply(lambda x: f"\\textbf{{{x}}}" if x == best else f"{x}")

        formatted = grouped.copy()
        for metric in metrics:
            minimise = metric in ("curr_steps", "cdt_subseq_steps")
            formatted[col_map[metric]] = bold_best(grouped[metric], minimise)

        # Reorder columns and format
        formatted = formatted[[col_map[metric] for metric in metrics]]
        formatted = formatted.reset_index()
        formatted = formatted.rename(columns={group_col: group_by.capitalize()})
        # formatted[group_col] = formatted[group_col].str.replace("_", r"\_")

        latex = formatted.to_latex(
            index=False,
            escape=False,
            column_format="l" + "c" * (len(formatted.columns) - 1),
            caption=f"Comparison of core metrics by {group_by}",
            label=f"tab:core_metrics_by_{group_by}",
        )

        with open(f"{self.tables_dir}/compact_metrics_core_table_by_{group_by}_new.tex", "w") as f:
            f.write(latex)

        return latex

    def core_metrics_across_tasks_latex_table(self, results, group_by="task"):
        df = results.copy()

        # Safe eval for stringified lists
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        metrics = [
            "curr_validity", "executability", "final_score", "trace_score", "curr_steps", "curr_last_act",
            "cdt_validity", "subseq_validity", "cdt_subseq_validity", "cdt_subseq_steps", "cdt_subseq_last_act",
            "improved_validity",
        ]
        col_map = self.get_column_names_map()

        for col in metrics:
            if df[col].dtype == object:
                print(col, df[col].unique()[:20])

        # Map the group_by option to the dataframe column
        group_col_map = {
            "task": "task_name",
            "model": "model_name",
            "domain": "domain",
        }
        if group_by not in group_col_map:
            raise ValueError(f"Unsupported group_by '{group_by}'. Choose from {list(group_col_map)}.")
        group_col = group_col_map[group_by]

        df = df[list(metrics) + [group_col]].copy()
        df = df.dropna()
        grouped = df.groupby(group_col).mean().round(2)

        # Bold best values (max, except for steps to validity: min)
        def bold_best(series, minimise=False):
            best = series.min() if minimise else series.max()
            return series.apply(lambda x: f"\\textbf{{{x}}}" if x == best else f"{x}")

        formatted = grouped.copy()
        for metric in metrics:
            minimise = metric in ("curr_steps", "cdt_subseq_steps")
            formatted[col_map[metric]] = bold_best(grouped[metric], minimise)

        # Reorder columns and format
        formatted = formatted[[col_map[metric] for metric in metrics]]
        formatted = formatted.reset_index()
        formatted = formatted.rename(columns={group_col: group_by.capitalize()})
        # formatted[group_col] = formatted[group_col].str.replace("_", r"\_")

        latex = formatted.to_latex(
            index=False,
            escape=False,
            column_format="l" + "c" * (len(formatted.columns) - 1),
            caption=f"Comparison of core metrics by {group_by}",
            label=f"tab:core_metrics_by_{group_by}",
        )

        with open(f"{self.tables_dir}/compact_metrics_core_table_by_{group_by}.tex", "w") as f:
            f.write(latex)

        return latex

    def get_performance_by_problem_difficulty(self, results, domain):
        domain_name = self.domain_name_mapping.get(domain, domain)
        # print(f"DEBUG: Using domain name '{domain_name}' for domain '{domain}'")
        goals_csv = os.path.join(results_dir, os.pardir, f"goal_counts/{domain_name}_goal_counts.csv") if domain_name else f"{results_dir}/../goal_counts/goal_counts.csv"
        goals_df = pd.read_csv(goals_csv, header=0)

        # self.plot_metric_by_group(results, goals_df, group_by="num_goals", domain=domain_name, metric="stv")
        self.plot_metric_by_group(results, goals_df, group_by="num_goals", domain=domain_name)
        # self.plot_metric_by_group(results, goals_df, group_by="num_goals", metric="LEA")
        # self.plot_metric_by_group(results, goals_df, group_by="gt_length", domain=domain_name, metric="stv")
        self.plot_metric_by_group(results, goals_df, group_by="gt_length", domain=domain_name)
        # self.plot_metric_by_group(results, goals_df, group_by="gt_length", metric="LEA")
        print(f"Performance by problem difficulty plots saved for domain '{domain_name}'.")

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

        if "domain" in results_df.columns:
            results_df["domain"] = results_df["domain"].apply(
                lambda x: self.domain_name_mapping.get(x, x)
            )

        df = pd.merge(results_df, goals_df, on=["domain", "instance_id"], how="inner")
        if domain:
            df = df[df["domain"] == self.domain_name_mapping.get(domain, domain)]

        # Add GT plan length if needed
        if not "optimal_plan" in df.columns:
            print(f"Warning: 'optimal_plan' column not found in results. Using 'gt_plan' for analysis.")
            df["optimal_plan"] = df["gt_plan"]

        df["gt_length"] = df["optimal_plan"].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x))
        df = df.dropna(subset=[group_by])

        for col in ["curr_validity", "curr_last_act", "curr_steps", "final_score", "trace_score", "gt_plan", "optimal_plan", "correct_part"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Drop rows where any metric is missing
        df = df.dropna(subset=["curr_validity", "curr_last_act", "curr_steps", "final_score", "trace_score"])

        # df["LEA"] = df["curr_last_act"] / df["gt_length"]
        # invalid_df = df[df["curr_validity"] == False]
        # lea_grouped = invalid_df.groupby(group_by)["LEA"].mean().reset_index(name="LEA/GT")

        grouped = df.groupby(group_by).agg({
            "curr_validity": "mean",
            # "LEA": "mean",
            "final_score": "mean",
            "curr_steps": "mean",
            # "trace_score": "mean",
            "instance_id": "count"
        }).rename(columns={
            "curr_validity": "SR",
            # "LEA": "LEA/GT",
            "final_score": "Score",
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
        plt.savefig(f"{self.figures_dir}/all_metrics_by_{group_by}_{domain}.pdf", format="pdf")
        plt.show()

    def plot_validity_steps_by_task(self, results):
        """
        Plots success rate and avg steps per plan type, grouped by task.
        """
        df = results.copy()

        for col in self.success_columns:
            df[col] = df[col].astype(bool)

        step_columns = [col.replace("validity", "steps") for col in self.success_columns]
        plan_types = [col.replace("_validity", "") for col in self.success_columns]

        tasks = df["task"].dropna().unique()
        n_tasks = len(tasks)

        fig, axs = plt.subplots(n_tasks, 1, figsize=(8, 3 * n_tasks), sharex=True)

        if n_tasks == 1:
            axs = [axs]

        for i, task in enumerate(sorted(tasks)):
            task_df = df[df["task"] == task]

            avg_success = [task_df[col].mean() * 100 for col in self.success_columns]
            avg_steps = [task_df[col].mean() for col in step_columns]

            ax1 = axs[i]
            color1, color2 = "royalblue", "seagreen"

            ax1.plot(plan_types, avg_success, marker="o", label="Success Rate (%)", color=color1)
            ax1.set_ylabel("Success (%)", color=color1)
            ax1.tick_params(axis="y", labelcolor=color1)

            ax2 = ax1.twinx()
            ax2.plot(plan_types, avg_steps, marker="s", label="Avg Steps", color=color2, linestyle="--")
            ax2.set_ylabel("Avg Steps", color=color2)
            ax2.tick_params(axis="y", labelcolor=color2)

            ax1.set_title(f"Task: {task}")
            ax1.grid(True, linestyle="--", alpha=0.5)

        axs[-1].set_xlabel("Plan Type")
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/validity_steps_by_task_lineplot.png", dpi=300)
        plt.show()

    def steps_validity_dual_axis_trends_by_plan(self, results):
        """
        Plots a line chart comparing avg steps (inverted) and success rate for each plan type (π₀ to πₙ).
        :param results: The dataframe to use.
        """

        df = results.copy()

        plan_labels = [f"$\\pi_{i}$" for i in range(len(self.success_columns))]
        step_columns = [col.replace("validity", "steps") for col in self.success_columns]

        data = []
        for label, valid_col, steps_col in zip(plan_labels, self.success_columns, step_columns):
            valid = df[valid_col].astype(bool)
            steps = df[steps_col]

            total = len(valid)
            success_rate = valid.sum() / total * 100
            avg_steps = steps.mean()

            data.append((label, avg_steps, success_rate))

        plot_df = pd.DataFrame(data, columns=["Plan", "avg_steps", "success_rate"])

        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Primary Y-axis (Avg Steps)
        ax1.set_xlabel("Plan Type", fontsize=18, labelpad=12)
        ax1.set_ylabel("Avg Steps to Validity", color="seagreen", fontsize=18, labelpad=12)
        ax1.plot(plot_df["Plan"], plot_df["avg_steps"], marker="s", label="Avg Steps to Validity", color="seagreen",
                 linewidth=2)
        ax1.tick_params(axis="y", labelcolor="seagreen", labelsize=14)
        ax1.tick_params(axis="x", labelsize=16)

        # Secondary Y-axis (Success Rate)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Success Rate (%)", color="darkorange", fontsize=18, labelpad=12)
        ax2.plot(plot_df["Plan"], plot_df["success_rate"], marker="^", label="Success Rate (%)", color="darkorange",
                 linewidth=2)
        ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=14)

        # Title and grid
        plt.title("Avg Steps and Success Rate per Plan Type", fontsize=20, pad=20)
        ax1.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2)
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/steps_validity_dual_axis_trends_by_plan.png", dpi=300)
        plt.show()

    def select_lengths(self, plan_lengths, n_lengths=4):
        """Pick n_lengths representative plan lengths from the observed distribution."""
        qs = np.linspace(0, 1, n_lengths)
        picked = np.quantile(plan_lengths, qs, method="nearest")
        return sorted(set(int(x) for x in picked))

    def extract_success_rates_at_lengths(self, results, validity_column_1, validity_column_2, plan_column=None, optimal_column="optimal_plan",
                                         lengths=None, n_lengths=8, min_count=1, group_by="optimal", verbose=True):
        """
        group_by="optimal":   bins = lengths of the optimal plan (problem difficulty)
        group_by="generated": bins = lengths of the LLM plan (requires plan_column)
        """
        df = results.copy()

        # df["optimal_plan_length"] = df[optimal_column].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x)).apply(len)
        df["optimal_plan"] = df[optimal_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["optimal_plan_length"] = df["optimal_plan"].apply(len)
        if plan_column is not None:
            # df["llm_plan_length"] = df[plan_column].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x)).apply(len)
            df["llm_plan_length"] = df["plan_length"]

        key = "optimal_plan_length" if group_by == "optimal" else "llm_plan_length"

        metrics = [validity_column_1, validity_column_2, "final_score", "curr_steps", "curr_last_act"]
        if plan_column is not None and key != "llm_plan_length":
            metrics.append("llm_plan_length")  # mean LLM plan length within each optimal-length bin

        grouped = df.groupby(key)[metrics].mean()
        counts = df[key].value_counts()

        if lengths is None:
            lengths = self.select_lengths(df[key], n_lengths)

        if verbose:
            print(f"{key} range: {df[key].min()}–{df[key].max()}, median {df[key].median():.0f}")
            print(f"Success Rates by {key}:")
            for length in lengths:
                n = counts.get(length, 0)
                if length in grouped.index and n >= min_count:
                    r = grouped.loc[length]
                    line = (f"Length {length} (n={n}): {validity_column_1} = {r[validity_column_1]:.2f}, "
                            f"{validity_column_2} = {r[validity_column_2]:.2f}, "
                            f"Score = {r['final_score']:.2f}, Steps = {r['curr_steps']:.2f}, "
                            f"LEA = {r['curr_last_act']:.2f}")
                    if "llm_plan_length" in r.index:
                        line += f", mean LLM plan len = {r['llm_plan_length']:.1f}"
                    print(line)
                else:
                    print(f"Length {length}: No data available")

        return grouped.loc[grouped.index.isin(lengths)]

    def success_rate_by_plan_length_two_plans(self, results, plan_column, validity_column_1, validity_column_2,
                                    label_1="Generated Plan", label_2="Most Improved Plan",
                                    title="Comparison of Success Rate by Plan Length"):
        """
        Compare the success rate by plan length for two types of plans.

        :param results: DataFrame containing the data
        :param plan_column: GT plan column
        :param validity_column_1: Plan validity column for the first plan type
        :param validity_column_2: Plan validity column for the second plan type
        :param label_1: Label for the first plan type (for legend)
        :param label_2: Label for the second plan type (for legend)
        :param title: Title of the plot
        """

        df = results.copy()
        df[plan_column] = df[plan_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Compute plan lengths
        df["plan_length"] = df[plan_column].apply(len)

        # Compute success rates per plan length for both validity metrics
        success_rate_1 = df.groupby("plan_length")[validity_column_1].mean()
        success_rate_2 = df.groupby("plan_length")[validity_column_2].mean()

        plt.figure(figsize=(8, 5))
        plt.plot(success_rate_1.index, success_rate_1.values, marker='o', linestyle='-', color='b', label=label_1)
        plt.plot(success_rate_2.index, success_rate_2.values, marker='s', linestyle='--', color='r', label=label_2)

        plt.xlabel("Plan Length (No. of Actions)", fontsize=18)
        plt.ylabel("Success Rate", fontsize=18)
        plt.title(title, fontsize=20)
        plt.ylim(0, 1)  # Success rate is between 0 and 1
        plt.xticks(success_rate_1.index, fontsize=16)  # Ensure all values are displayed
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(f"{self.figures_dir}/success_rate_by_plan_length.png", dpi=300)
        plt.show()

    def plan_length_frequency(self, results, valid_only=False, plan_column="curr_plan", validity_column="curr_validity"):
        """
        Plots the distribution of plan lengths as a percentage of total plans.
        :param results: The dataframe containing the plans.
        :param valid_only: If True, only considers valid plans (curr_validity == True).
        :param plan_column: The column containing the plans.
        :param validity_column: The column containing the validity of the plans.
        """
        df = results.copy()

        if valid_only:
            df = df[df[validity_column] == True]

        df["plan_length"] = df[plan_column].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else len(x))
        length_counts = df["plan_length"].value_counts(normalize=True) * 100  # Convert to percentage

        # Sort by plan length for ordered bars
        length_counts = length_counts.sort_index()

        plt.figure(figsize=(10, 5))
        plt.bar(length_counts.index, length_counts.values, color="royalblue", alpha=0.7)
        plt.xlabel("Plan Length")
        plt.ylabel("Percentage of Total Plans (%)")

        title = "Distribution of Plan Lengths"
        if plan_column == "cdt_plan":
            title += ": Candidate Plan"
        elif plan_column == "subseq_plan":
            title += ": Improved Plan"
        elif plan_column == "cdt_subseq_plan":
            title += ": Improved Candidate Plan"
        elif plan_column == "gt_plan":
            title += ": Ground Truth Plan"

        if valid_only:
            if validity_column == "executability":
                title += " (Executable Plans Only)"
            else:
                title += " (Valid Plans Only)"

        plt.title(title)
        plt.xticks(length_counts.index)  # Ensure all values are displayed
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        plt.show()

    def normalised_metrics_per_instance_id_with_ci(self, results, plan_column, last_exec_column, score_column, validity_column):
        df = results.copy()

        # Convert lists from strings and get length of plan
        df[plan_column] = df[plan_column].apply(eval)
        df["plan_length"] = df[plan_column].apply(len)

        # Normalise metrics
        df["normalised_last_exec"] = (df[last_exec_column] / df["plan_length"]) * 100  # % of plan executed
        df["normalised_score"] = df[score_column].apply(lambda x: max(0, (x / 20) * 100))  # Score as %, capped at 0% and max score is 20
        df["normalised_validity"] = df[validity_column] * 100  # Convert True/False to %

        # Group by instance_id
        instance_column = "instance_id"
        grouped = df.groupby(instance_column)[["normalised_last_exec", "normalised_score", "normalised_validity"]]
        mean_metrics = grouped.mean().reset_index()
        std_metrics = grouped.std().reset_index()
        count_metrics = grouped.count().reset_index()  # Count of experiments per instance

        # Compute 95% Confidence Interval (CI = 1.96 * (std / sqrt(n)))
        ci_95 = 1.96 * (std_metrics / np.sqrt(count_metrics)).reset_index()

        plt.figure(figsize=(10, 6))
        for metric, label, color in zip(
                ["normalised_last_exec", "normalised_score", "normalised_validity"],
                ["Last Executable Action (%)", "Score (%)", "Validity (%)"],
                ["blue", "green", "red"]
        ):
            plt.plot(mean_metrics[instance_column], mean_metrics[metric], marker='o', linestyle='-', label=label,
                     color=color)

            # Add shaded 95% CI
            plt.fill_between(
                mean_metrics[instance_column],
                mean_metrics[metric] - ci_95[metric],
                mean_metrics[metric] + ci_95[metric],
                color=color,
                alpha=0.2
            )

        plt.xlabel("Instance ID")
        plt.ylabel("Percentage (%)")
        plt.title("Normalised Metrics per Instance (Mean ± 95% CI)")
        plt.legend(title="Metrics")
        plt.ylim(0, 100)  # All metrics are percentages (0-100%)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

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

    def plot_percentage_distribution_with_lines(self, results, cap=-1, plan_column="plan_trace"):
        """
        Plot the distribution of values across indices, as a percentage normalised by the number of occurrences
        of each index. Also computes Spearman correlation between index and label percentages.
        If cap is -1, use the 95th percentile index value as a cap.
        """
        df = results.copy()

        values, column_name, df = self.map_values_to_scores(df, plan_column)
        if plan_column == "nr_plan_trace":
            possible_values = ["correct", "same_act", "diff_act", "redundant"]
        else:
            possible_values = ["correct", "same_act", "diff_act", "redundant", "misplaced"]

        # Determine dynamic cap if cap == -1
        all_indices_flat = []
        for dict_data in df[column_name]:
            if isinstance(dict_data, dict):
                all_indices_flat.extend(dict_data.keys())
        if cap == -1 and all_indices_flat:
            all_indices_flat = np.array(all_indices_flat).astype(int)
            cap = int(np.percentile(all_indices_flat, 95))
            # print(f"Capping at 95th percentile: index {cap}")

        value_counts = {}
        index_occurrences = {}

        for dict_data in df[column_name]:
            if isinstance(dict_data, dict):
                for index, value in dict_data.items():
                    if index >= cap:
                        continue
                    if plan_column == "nr_plan_trace" and value == "misplaced": # shouldn't happen anymore
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
        values_matrix = {value: [percentage_distribution[idx].get(value, 0) for idx in sorted_indices] for value in
                         possible_values}

        # Map qualitative labels to ordinal values
        label_to_score = {
            "correct": 4,
            "misplaced": 3,
            "same_act": 2,
            "diff_act": 1,
            "redundant": 0,
        }

        all_indices = []
        all_scores = []

        for dict_data in df[column_name]:
            if isinstance(dict_data, dict):
                for index, label in dict_data.items():
                    if index >= cap:
                        continue
                    if plan_column == "nr_plan_trace" and label == "misplaced":
                        label = "correct"
                    score = label_to_score.get(label)
                    if score is not None:
                        all_indices.append(index)
                        all_scores.append(score)

        if all_indices:
            rho, p = spearmanr(all_indices, all_scores)
            print(f"\nOverall Spearman correlation between index and label score: rho = {rho:.2f}, p = {p:.4f}")
        else:
            rho, p = None, None
            print("No valid data to compute overall correlation.")

        plt.figure(figsize=(10, 5))
        for value in possible_values:
            plt.plot(sorted_indices, values_matrix[value], marker='o', linestyle='-', label=value)

        plt.xlabel("Index", fontsize=20)
        plt.ylabel("Percentage (%)", fontsize=20)

        # title = f"NP-AQM" if plan_column == "nr_plan_trace" else "AQM"
        # title += f": first {cap} indices; Spearman: $\\rho={rho:.2f}, $p$={p:.4f}$"
        title = f"Spearman: $\\rho={rho:.2f}, $p$={p:.4f}$"
        plt.title(title, fontsize=22)

        filename = f"{self.figures_dir}/percentage_distribution_with_lines_{plan_column}_{cap}.pdf"
        plt.legend(title="Value", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(sorted_indices, fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, format="pdf", dpi="figure")
        # plt.show()

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

        print(f"\nHigh SR Spearman: $\\rho={rho_h:.2f}, p={p_h:.4f}$, plan_column={plan_column}")
        print(f"Low SR Spearman: $\\rho={rho_l:.2f}, p={p_l:.4f}$")

        # Compact legend components
        legend_elements = [
                              Line2D([0], [0], color='black', linestyle='-', label='High SR'),
                              Line2D([0], [0], color='black', linestyle='--', label='Low SR')
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

        filename = f"{self.figures_dir}/percentage_distribution_with_lines_{plan_column}_compare_{cap_used}_high_and_low.pdf"
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
    def last_exec_frequency_one_plan(self, results, last_exec_column):
        """
        Plot the frequency of each last executable action in a single plan
        :param df: the dataframe to use
        :param last_exec_column: the column containing the last executable action
        :return:
        """
        df = results.copy()

        value_counts = df[last_exec_column].value_counts(normalize=True) * 100

        # value_counts = value_counts.sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(value_counts.index, value_counts.values, color="blue", alpha=0.7)
        plt.xlabel("Last Executable Action Value")
        plt.ylabel("Frequency (%)")
        plt.title("Distribution of Last Executable Actions")
        plt.xticks(value_counts.index, rotation=0)  # Rotate x labels if many unique values
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        plt.show()

    def last_exec_frequency_relative_to_length(self, results, last_exec_column, plan_column, group=5):
        """
        Plot the frequency of each last executable action as a percentage of the total plan length,
        grouped into percentage levels.

        :param results: The dataframe to use
        :param last_exec_column: The column containing the last executable action index
        :param plan_column: The column containing the full plan (stored as a list in CSV)
        :param group: The size of the percentage bins (default is 5%)
        """
        df = results.copy()

        # Convert plan from string to list if needed
        # df[plan_column] = df[plan_column].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df[plan_column] = df[plan_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Filter out empty plans
        for col in [plan_column, "gt_plan"]:
            if col in df.columns:
                df = df[df[col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

        # df = df[df[plan_column].apply(lambda x: len(x) > 0)]

        # Compute relative percentage position
        def compute_relative_position(row):
            if len(row[plan_column]) == 0:
                return 0  # Set to 0% if the plan is empty
            return ((row[last_exec_column]) / len(row[plan_column])) * 100

        df["relative_position"] = df.apply(compute_relative_position, axis=1)

        bins = list(range(0, 101, group))
        labels = [f"{i}-{i + group}%" for i in bins[:-1]]

        df["grouped_position"] = pd.cut(df["relative_position"], bins=bins, labels=labels, include_lowest=True)
        value_counts = df["grouped_position"].value_counts(normalize=True) * 100
        value_counts = value_counts.sort_index()

        # Plot
        plt.figure(figsize=(9, 5))
        plt.bar(value_counts.index, value_counts.values, color="blue", alpha=0.7, width=0.7)
        plt.xlabel("Portion of executable part from full $\\pi_{\\text{LLM}}$", fontsize=18)
        plt.ylabel("Frequency (%)", fontsize=18)
        # plt.title("Relative Position of Last Executable Action in Plans", fontsize=20)
        plt.xticks(rotation=45, ha="right", fontsize=16)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/last_exec_frequency_relative_to_length.pdf", dpi="figure", format="pdf")
        # plt.show()

    def last_exec_frequency_relative_to_length_high_and_low(self, last_exec_column, plan_column, group=5):
        """
        Plot the frequency of each last executable action as a percentage of the total plan length,
        grouped into percentage levels, comparing high and low success models.
        """
        # Prepare the datasets
        value_counts_high = self.prepare_data(self.high_success, last_exec_column, plan_column, group)
        value_counts_low = self.prepare_data(self.low_success, last_exec_column, plan_column, group)

        # Plot
        plt.figure(figsize=(10, 5))
        x = np.arange(len(value_counts_high.index))
        width = 0.35

        plt.bar(x - width / 2, value_counts_high.values, color="blue", alpha=0.45, width=width, label="High SR")
        plt.bar(x + width / 2, value_counts_low.values, color="blue", alpha=0.8, width=width, label="Low SR")

        plt.xlabel("Portion of executable part from full $\\pi_{\\text{LLM}}$", fontsize=26)
        plt.ylabel("Frequency (%)", fontsize=26)
        plt.xticks(x, value_counts_high.index, rotation=45, ha="right", fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(fontsize=18)

        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/last_exec_frequency_relative_to_length_h_and_l.pdf", dpi="figure", format="pdf")
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
            return (row[last_exec_column]) * 100 # / len(row[plan_column]))

        df["relative_position"] = df.apply(compute_relative_position, axis=1)

        bins = list(range(0, 101, group))
        labels = [f"{i}-{i + group}%" for i in bins[:-1]]

        df["grouped_position"] = pd.cut(df["relative_position"], bins=bins, labels=labels, include_lowest=True)
        value_counts = df["grouped_position"].value_counts(normalize=True) * 100
        value_counts = value_counts.sort_index()
        return value_counts

    # No REASONING
    def complemenary_length_out_of_gt_length(self, results):
        """
        Plots the mean proportion of complementary_plan length relative to gt_plan length as a smooth line
        with variance shading, grouped by last executable action.
        It includes both cases: with and without last executable action.

        :param results: The dataframe to use.
        """
        df = results.copy()

        df["gt_plan"] = df["gt_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["complementary_plan"] = df["complementary_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["improved_plan"] = df["improved_plan"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        df["gt_length"] = df["gt_plan"].apply(len)
        df["complementary_length"] = df["complementary_plan"].apply(len)
        df["improved_length"] = df["improved_plan"].apply(len)
        df["correct_part_length"] = df["correct_part"].apply(len)

        # Filter to only where a complementary plan was generated # and a correct part exists (other cases are: goal in plan, or GT had to be generated)
        df = df[df["complementary_length"] > 0]
        df = df[df["correct_part_length"] > 0]

        # Compute both ratios with and without last executable action
        df["length_ratio_no_exec"] = df["complementary_length"] / df["gt_length"]
        df["length_ratio_with_exec"] = (df["complementary_length"] + df["curr_last_act"]) / df["gt_length"]
        df["length_ratio_correct_part"] = df["correct_part_length"] / df["gt_length"]

        # Cap curr_last_act at 99th percentile to avoid long-tail noise
        cap_value = int(np.percentile(df["curr_last_act"], 99.5))
        df = df[df["curr_last_act"] <= cap_value]

        # Group by last executable action for smoothing
        grouped_df_no_exec = df.groupby("curr_last_act")["length_ratio_no_exec"].agg(["mean", "std"]).reset_index()
        grouped_df_with_exec = df.groupby("curr_last_act")["length_ratio_with_exec"].agg(["mean", "std"]).reset_index()
        grouped_df_correct_part = df.groupby("curr_last_act")["length_ratio_correct_part"].agg(["mean", "std"]).reset_index()

        # Plot
        plt.figure(figsize=(9, 5))

        # Without last executable
        plt.plot(grouped_df_no_exec["curr_last_act"], grouped_df_no_exec["mean"],
                 color="royalblue", marker="o", linestyle="-", label="$\\pi_{\\text{comp}}$ / $\\pi_{\\text{GT}}$ Ratio")
        plt.fill_between(grouped_df_no_exec["curr_last_act"],
                         grouped_df_no_exec["mean"] - grouped_df_no_exec["std"],
                         grouped_df_no_exec["mean"] + grouped_df_no_exec["std"],
                         color="royalblue", alpha=0.3)

        # Correct part only
        plt.plot(grouped_df_correct_part["curr_last_act"], grouped_df_correct_part["mean"],
                    color="green", marker="^", linestyle=":", label="$\\pi_{\\text{corr}}$ / $\\pi_{\\text{GT}}$ Ratio")
        plt.fill_between(grouped_df_correct_part["curr_last_act"],
                            grouped_df_correct_part["mean"] - grouped_df_correct_part["std"],
                            grouped_df_correct_part["mean"] + grouped_df_correct_part["std"],
                            color="green", alpha=0.3)

        plt.xlabel("Last Executable Action Index", fontsize=20)
        plt.ylabel("Portion from $\\pi_{\\text{GT}}$", fontsize=20)
        # plt.title("Comparison of contribution: LLM vs. Planner", fontsize=20)
        plt.ylim(0, 2)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/complementary_length_out_of_gt_length.pdf", format="pdf", dpi="figure")
        # plt.show()

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

        plt.savefig(f"{self.figures_dir}/complementary_length_out_of_gt_length_high_and_low.pdf", format="pdf",
                    dpi="figure")
        print(f"Saved complementary length plot for high and low success models to {self.figures_dir}/complementary_length_out_of_gt_length_high_and_low.pdf")
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

        # Per-instance group
        # df = self.abbreviate_config_names(df)
        instance_df = df.groupby(["instance_id", f"{group_by_column}_name_abbv"]).agg({
            executability_column: "mean",
            validity_column: "mean"
        }).reset_index()

        # Group by model/task
        grouped = instance_df.groupby(f"{group_by_column}_name_abbv")
        plot_df = grouped.agg(
            Executability=(executability_column, 'mean'),
            Validity=(validity_column, 'mean'),
            Executability_std=(executability_column, 'std'),
            Validity_std=(validity_column, 'std')
        ).reset_index()

        # Convert to %
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


        # Plot
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
        model_names = plot_df[f"{group_by_column}_name_abbv"]
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="center", fontsize=13)

        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/executability_validity_stacked.pdf", format="pdf", dpi="figure")
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
            abbv_cols = df[
                ["domain", "model", "task", "domain_name_abbv", "model_name_abbv", "task_name_abbv"]].drop_duplicates()
            grouped = pd.merge(grouped, abbv_cols, on=["domain", "model", "task"], how="left")
            # grouped = grouped[grouped["count"] >= 10]

            # grouped["model"] = grouped["model"].apply(self.map_model_names)
            return grouped

        def print_stats(grouped, label):
            print(f"=== LEA vs SR: {label.upper()} MODELS ===")
            quantiles = pd.qcut(grouped["success_rate"], q=3, duplicates="drop", retbins=True)
            num_bins = len(quantiles[1]) - 1
            label_options = ["Low", "Medium", "High"]
            grouped["success_group"] = pd.qcut(grouped["success_rate"], q=3, labels=label_options[:num_bins], duplicates="drop")

            low = grouped[grouped["success_group"] == "Low"]["mean_last_exec"]
            med = grouped[grouped["success_group"] == "Medium"]["mean_last_exec"]
            high = grouped[grouped["success_group"] == "High"]["mean_last_exec"]
            H, p_kw = kruskal(low, med, high)
            print(f"Kruskal-Wallis H = {H:.2f}, p = {p_kw:.4f}")

            lea_quantiles = pd.qcut(grouped["mean_last_exec"], q=3, duplicates="drop", retbins=True)
            num_lea_bins = len(lea_quantiles[1]) - 1
            lea_labels = ["Short", "Medium", "Long"][:num_lea_bins]
            grouped["lea_bin"] = pd.qcut(grouped["mean_last_exec"], q=3, labels=lea_labels, duplicates="drop")

            contingency = pd.crosstab(grouped["success_group"], grouped["lea_bin"])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p_chi2, _, _ = chi2_contingency(contingency)
                print(f"Chi-Square χ² = {chi2:.2f}, p = {p_chi2:.4f}")
            else:
                print("Chi-Square test not applicable due to insufficient contingency table size.")

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

        # Stats
        rho_low, p_low = print_stats(low_grouped, "Low SR")
        rho_high, p_high = print_stats(high_grouped, "High SR")
        rho_total, p_total = print_stats(results_grouped, "Total SR")

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, grouped, title, rho, p in zip(
                axs,
                [low_grouped, high_grouped],
                ["Low SR Models", "High SR Models"],
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

            # grouped = self.abbreviate_config_names(grouped.copy())

            for _, row in grouped.iterrows():
                x_jitter = np.random.uniform(-0.01, 0.01)
                y_jitter = np.random.uniform(-0.2, 0.2)
                label_str = f"{row["model_name_abbv"]}/{row["task_name_abbv"]}" if (self.domain and self.domain != "baseline") else f"{row["domain_name_abbv"]}/{row["model_name_abbv"]}/{row["task_name_abbv"]}"
                ax.text(row["success_rate"] + x_jitter, row["mean_last_exec"] + y_jitter, label_str, fontsize=8, alpha=0.8)

            ax.set_title(f"{title}; Spearman ρ = {rho:.2f}, p = {p:.4f}", fontsize=20)
            ax.set_xlabel("Success Rate", fontsize=20)
            ax.grid(True, linestyle="--", alpha=0.5)

        axs[0].set_ylabel("Mean LEA for Invalid Plans", fontsize=20)
        axs[1].set_ylabel("")

        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/success_vs_last_executable_high_and_low.pdf", format="pdf", dpi="figure")
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
            ("Blocksworld 3", "Ground Truth"): "#ff7f0e",
            ("Blocksworld 3", "LLM-Generated"): "#ffbb78",
            ("Logistics", "Ground Truth"): "#ff7f0e",
            ("Logistics", "LLM-Generated"): "#ffbb78",
            ("Mystery BW", "Ground Truth"): "#2ca02c",
            ("Mystery BW", "LLM-Generated"): "#98df8a",
            ("Random BW", "Ground Truth"): "#d62728",
            ("Random BW", "LLM-Generated"): "#ff9896",
            ("Random Logistics", "Ground Truth"): "#9467bd",
            ("Random Logistics", "LLM-Generated"): "#c5b0d5",
            ("Obfuscated Random Logistics", "Ground Truth"): "#8c564b",
            ("Obfuscated Random Logistics", "LLM-Generated"): "#c49c94",
            ("Obfuscated Deceptive Logistics", "Ground Truth"): "#698b22",
            ("Obfuscated Deceptive Logistics", "LLM-Generated"): "#4c4c4c",
            ("Elevator", "Ground Truth"): "#e377c2",
            ("Elevator", "LLM-Generated"): "#f7b6d2",
            ("Satellite", "Ground Truth"): "#7f7f7f",
            ("Satellite", "LLM-Generated"): "#c7c7c7",
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
        plt.savefig(f"{self.figures_dir}/plan_length_distributions_by_domain.pdf", format="pdf", dpi="figure")
        plt.show()

    def map_domain_names(self, domain):
        """
        Maps domain names to their full names for better readability in plots.
        :param domain_list: List of domain names.
        :return: List of mapped domain names.
        """
        domain_map = {
            "blocksworld": "Blocksworld",
            "blocksworld_3": "Blocksworld 3",
            "mystery_blocksworld": "Mystery BW",
            "random_blocksworld": "Random BW",
            "elevator": "Elevator",
            "logistics": "Logistics",
            "satellite": "Satellite",
            "obfuscated_deceptive_logistics": "Obfuscated Deceptive Logistics",
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

        df["curr_validity"] = df["curr_validity"].astype(float)
        df["cdt_validity"] = df["cdt_validity"].astype(float)
        df["subseq_validity"] = df["subseq_validity"].astype(float)
        df["cdt_subseq_validity"] = df["cdt_subseq_validity"].astype(float)
        df["improved_validity"] = df["improved_validity"].astype(float)

        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        # Metrics and column names
        metrics = [
            "curr_validity", "final_score", "trace_score", "curr_steps", "correct_part_len", "curr_last_act",
            "cdt_validity", "cdt_last_act", "subseq_validity", "cdt_subseq_validity", "cdt_subseq_steps", "improved_validity"
        ]
        col_map = self.get_column_names_map()

        latex_table = self.format_latex_table(
            df, metrics, col_map,
            "Mean values of evaluation metrics per experiment (model × task). "
            "Bold highlights the best value per model.",
            f"tab:appendix_metrics_per_experiment",
            f"{self.tables_dir}/appendix_metrics_per_experiment.tex"
        )

    def latex_one_summary_table_per_experiment_by_domain(self, results):
        """
        Creates a LaTeX table for appendix, grouped by domain x model × task.
        Improves formatting with bolded best values per model, model-task column split, and midrules.
        """
        df = results.copy()
        df["correct_part"] = df["correct_part"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        df["curr_validity"] = df["curr_validity"].astype(float)
        df["cdt_validity"] = df["cdt_validity"].astype(float)
        df["subseq_validity"] = df["subseq_validity"].astype(float)
        df["cdt_subseq_validity"] = df["cdt_subseq_validity"].astype(float)
        df["improved_validity"] = df["improved_validity"].astype(float)

        df["correct_part_len"] = df["correct_part"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

        # Metrics and column names
        metrics = [
            "curr_validity", "final_score", "trace_score", "curr_steps", "correct_part_len", "curr_last_act",
            "cdt_validity", "cdt_last_act", "subseq_validity", "cdt_subseq_validity", "cdt_subseq_steps", "improved_validity"
        ]
        col_map = self.get_column_names_map()

        latex_table = self.format_latex_table(
            df, metrics, col_map,
            "Mean values of evaluation metrics per experiment (domain x model × task). "
            "Bold highlights the best value per model.",
            f"tab:appendix_metrics_per_experiment_by_domain",
            f"{self.tables_dir}/appendix_metrics_per_experiment_by_domain.tex",
            group_by=["domain_name", "model_name", "task_name"]
        )

    ### HELPER functions
    def format_latex_table(self, df, metrics, col_map, caption, label, file_path, group_by=["model_name", "task_name"]):
        summary = df.groupby(group_by)[metrics].mean().reset_index()

        # Build LaTeX table
        lines = []
        groups: str = "\\small Model & \\small Task & " if group_by == ["model_name", "task_name"] else "\\small Domain & \\small Model & \\small Task & "
        header = (
                "\\begin{table}[ht]\n\\centering\n"
                f"\\caption{{{caption}}}\n"
                f"\\label{{{label}}}\n"
                "\\resizebox{\\textwidth}{!}{%\n"
                "\\begin{tabular}{ll" + "c" * len(metrics) + "}\n"
                                                             "\\toprule\n" + groups + " & \\small ".join(
            [f"\\textbf{{{col_map[col]}}}" for col in metrics]) + " \\\\\n"
                                                                  "\\midrule"
        )
        lines.append(header)

        for model in summary["model_name"].unique():
            model_df = summary[summary["model_name"] == model].copy()
            # Columns where lower is better
            minimise_cols = {"curr_steps", "cdt_subseq_steps"}

            bold_vals = {}
            for col in metrics:
                if col in minimise_cols:
                    bold_vals[col] = model_df[col].min()
                else:
                    bold_vals[col] = model_df[col].max()

            for _, row in model_df.iterrows():
                if group_by == ["model_name", "task_name"]:
                    line = f"{row['model_name']} & {row['task_name']}"
                else:
                    line = f"{row['domain_name']} & {row['model_name']} & {row['task_name']}"
                for col in metrics:
                    val = row[col]
                    if pd.isna(val):
                        formatted = "--"
                    elif val == bold_vals[col]:
                        formatted = f"\\textbf{{{val:.2f}}}"
                    else:
                        formatted = f"{val:.2f}"
                    line += f" & {formatted}"
                line += " \\\\"
                lines.append(line)

            lines.append("\\hline")

        footer = "\\bottomrule\n\\end{tabular}%\n}\\end{table}"
        lines.append(footer)

        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        return "\n".join(lines)

    def get_column_names_map(self):
        return {
            "curr_validity": "$\\pi_{\\text{LLM}}$ SR $\\uparrow$",
            "executability": "$\\pi_{\\text{LLM}}$ ER $\\uparrow$",
            "curr_score": "S $\\uparrow$",
            "trace_score": "AQM $\\uparrow$",
            "curr_steps": "StV $\\downarrow$",
            "correct_part_len": "len($\\pi_{\\text{corr}}$ $\\uparrow$",
            "curr_last_act": "LEA $\\uparrow$",
            "cdt_validity": "$\\pi_{\\text{NLP}}$ SR $\\uparrow$",
            "cdt_last_act": "$\\pi_{\\text{LCS}}$ LEA $\\uparrow$",
            "final_score": "Score $\\uparrow$",
            "final_score_validity": "ScoreN $\\uparrow$",
            "subseq_validity": "$\\pi_{\\text{LCS}}$ SR $\\uparrow$",
            "cdt_subseq_validity": "$\\pi_{\\text{NLP-LCS}}$ SR $\\uparrow$",
            "cdt_subseq_steps": "$\\pi_{\\text{NLP-LCS}}$ StV $\\downarrow$",
            "cdt_subseq_last_act": "$\\pi_{\\text{NLP-LCS}}$ LEA",
            "improved_validity": "$\\pi_{\\text{VAL}}$ SR $\\uparrow$",
        }

    def map_model_names(self, models):
        """
        Maps model names to more readable versions.
        :param models: List of model names.
        :return: New list with mapped names.
        """
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
            "gemma4-12b": "Gemma 4 12B",
            "gemma4-26b-a4b_it": "Gemma 4 26B",
            "qwen3-8b": "Qwen3 8B",
            "qwen3-14b": "Qwen3 14B",
            "qwen3-30b-a3b": "Qwen3 30B A3B",
            # "dsqwen": "DeepSeek Qwen",
        }
        for i in range(len(models)):
            if models[i] in model_map:
                models[i] = model_map[models[i]]
        return models

    def save_row_to_text(self, csv_filename, row_index):
        """
        Extracts a specific row from a CSV file and saves it as a text file.

        :param csv_filename: The path to the CSV file.
        :param row_index: The index of the row to extract.
        """
        df = pd.read_csv(csv_filename)
        output_filename = "outputs/running_example.txt"

        # Check if the row index is valid
        if row_index < 0 or row_index >= len(df):
            print("Invalid row index")
            return

        row_data = df.iloc[row_index].to_dict()
        formatted_data = "\n".join([f"{key}: {value}" for key, value in row_data.items()])

        with open(output_filename, "w") as f:
            f.write(formatted_data)

        print(f"Row {row_index} saved to {output_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse plan recovery results")
    parser.add_argument("--missing_instances", action="store_true", help="Whether to include missing instances in the analysis")
    parser.add_argument("--domain", type=str, default="", choices=["", "baseline", "blocksworld_3", "blocksworld", "logistics", "elevator", "satellite",  "mystery_bw", "random_bw", "deceptive_lg", "random_lg"], help="Domain to filter results by (blocksworld or logistics)")
    parser.add_argument("--success_rate", type=str, default="", choices=["high_success", "low_success", ""],
                        help="Success level to filter results by (high_success, low_success, or all)")
    # parser.add_argument("--LRMs", alction="store_true", default=True, help="Whether to evaluate LLMs or LRMs (default: LLMs)")
    parser.add_argument("--eval_config", type=str, default="", choices=["", "_only_optimal_gt", "_search_params", "_baseline"], help="Evaluation configuration to use (default: _only_optimal_gt)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    missing_instances = args.missing_instances
    domain = args.domain  # "blocksworld" or "logistics" ("" for all domains)
    success_level = args.success_rate
    # LRMs = args.LRMs  # True for LRMs, False for LLMs

    current_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    # eval_config = "_only_optimal_gt"
    eval_config = args.eval_config
    results_dir = f"{project_dir}/results/plan_recovery/final_evaluation{eval_config}/"
    # results_dir = f"{project_dir}/results/plan_recovery/final_evaluation/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if domain:
        missing_instances_file = f"{results_dir}/full_evaluation_w_missing_instances_{domain}.csv"
    else:
        missing_instances_file = f"{results_dir}/full_evaluation_w_missing_instances.csv"

    # suffix = "_LRMs" if LRMs else ""
    suffix = f"_{success_level}" if success_level else ""
    # suffix = "discard_empty"

    results_dirname = f"final_evaluation{eval_config}{suffix}"
    if missing_instances:
        results_file = missing_instances_file
        if domain:
            figures_dir = f"{project_dir}/figures/plan_recovery/{results_dirname}/figures_w_missing_instances_{domain}"
            tables_dir = f"{project_dir}/tables/plan_recovery/{results_dirname}/tables_w_missing_instances_{domain}"
        else:
            figures_dir = f"{project_dir}/figures/plan_recovery/{results_dirname}/figures_w_missing_instances"
            tables_dir = f"{project_dir}/tables/plan_recovery/{results_dirname}/tables_w_missing_instances"
    else:
        if domain:
            results_file = f"{results_dir}/full_evaluation_{domain}.csv"
            figures_dir = f"{project_dir}/figures/plan_recovery/{results_dirname}/figures_{domain}"
            tables_dir = f"{project_dir}/tables/plan_recovery/{results_dirname}/tables_{domain}"
        else:
            results_file = f"{results_dir}/full_evaluation.csv"
            figures_dir = f"{project_dir}/figures/plan_recovery/{results_dirname}/figures_full"
            tables_dir = f"{project_dir}/tables/plan_recovery/{results_dirname}/tables_full"
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

    # Check if results already post-processed, and use that
    post_processed = results_file.replace(".csv", "_postprocessed.csv")
    is_processed = False
    if os.path.exists(post_processed):
        print(f"Post-processed results file already exists: {post_processed}, using that instead")
        results_file = post_processed
        is_processed = True

    results = pd.read_csv(results_file)
    # TODO still need the whole missing instances situation??
    # if not missing_instances and not os.path.exists(missing_instances_file):
    #     fill_missing_instances(results, missing_instances_file)

    # results_dir = os.path.join(results_dir, os.pardir) if "o4-mini" in domain else results_dir
    analyser = AnalyseResults(results_file, results_dir, is_processed=is_processed, success_level=success_level, domain=domain, figures_dir=figures_dir, tables_dir=tables_dir)
    # analyser.save_row_to_text(f"{results_dir}/blocksworld_3_evaluation_qcode_one_shot.csv", 8) # QCODE one-shot instance 10