import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

from analyse_results import AnalyseResults

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

MODEL_FAMILY_KEYWORDS = {
    "qwen": "qwen",
    "gemma": "gemma",
    "gemini": "gemini",
    "claude": "claude",
    "gpt": "openai",
}
MODEL_FAMILY_PREFIXES = {
    "o3": "openai",
    "o4": "openai",
}

class CompareResults:
    def __init__(self, project_dir, domain=None, success_level=None, missing_instances=False, tables_dir=None, figures_dir=None):
        new_file, new_dir, new_processed = resolve_results_file(project_dir, "", domain, missing_instances)
        old_file, old_dir, old_processed = resolve_results_file(project_dir, "_old", domain, missing_instances)

        self.new = AnalyseResults(new_file, new_dir, is_processed=new_processed, success_level=success_level, domain=domain, run_analysis=False)
        self.old = AnalyseResults(old_file, old_dir, is_processed=old_processed, success_level=success_level, domain=domain, run_analysis=False)

        self.df_new = self.new.results
        # Filter new results to only include blocksworld_3 and logistics domains
        self.df_new = self.df_new[self.df_new["domain"].isin(["blocksworld_3", "logistics"])]
        self.df_old = self.old.results

        self.tables_dir = tables_dir
        self.figures_dir = figures_dir
        self.add_model_family_column()

        self.run_analysis()

    def run_analysis(self):
        self.compare_lea_by_domain()
        self.compare_sr_by_domain()

        self.compare_validity_improvement()
        self.validity_improvement_standalone(group_by="model_name")

        self.compare_success_rates_at_lengths()

        # self.compare_success_rates_at_lengths_graph(split_by="domain")
        # self.compare_success_rates_at_lengths_graph(split_by="model_name")
        # self.compare_success_rates_at_lengths_graph(split_by="model_family")
        #
        # self.compare_success_rates_at_lengths_graph(split_by="domain", group_by="curr_plan")
        # self.compare_success_rates_at_lengths_graph(split_by="model_name", group_by="curr_plan")
        # self.compare_success_rates_at_lengths_graph(split_by="model_family", group_by="curr_plan")



    # LEA per domain
    def compare_lea_by_domain(self, lea_column="curr_last_act"):
        common_domains = sorted(
            set(self.df_old["domain"].unique()) & set(self.df_new["domain"].unique())
        )
        if not common_domains:
            raise ValueError("No overlapping domains found between old and new results.")

        old_lea = self.df_old[self.df_old["domain"].isin(common_domains)].groupby("domain")[lea_column].mean()
        new_lea = self.df_new[self.df_new["domain"].isin(common_domains)].groupby("domain")[lea_column].mean()

        table = pd.DataFrame({"old_lea": old_lea, "new_lea": new_lea})
        table["change"] = table["new_lea"] - table["old_lea"]
        table = table.round(2)

        print("=== LEA by domain (old vs new) ===")
        print(table)

        fig, ax = plt.subplots(figsize=(6, 4))
        table[["old_lea", "new_lea"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean LEA")
        ax.set_title("LEA by Domain: Old vs New")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        if self.figures_dir:
            plt.savefig(f"{self.figures_dir}/lea_by_domain_old_vs_new.pdf")
        plt.show()

        return table

    def compare_sr_by_domain(self, validity_col="curr_validity", improved_col="cdt_subseq_validity"):
        common_domains = sorted(
            set(self.df_old["domain"].unique()) & set(self.df_new["domain"].unique())
        )
        if not common_domains:
            raise ValueError("No overlapping domains found between old and new results.")

        old_sr = self.df_old[self.df_old["domain"].isin(common_domains)].groupby("domain").agg(
            overall_success_rate=(validity_col, "mean"),
            improved_success_rate=(improved_col, "mean")
        ).loc[common_domains]
        new_sr = self.df_new[self.df_new["domain"].isin(common_domains)].groupby("domain").agg(
            overall_success_rate=(validity_col, "mean"),
            improved_success_rate=(improved_col, "mean")
        ).loc[common_domains]

        old_color = "tab:blue"
        new_color = "tab:orange"

        x = np.arange(len(common_domains))
        bar_width = 0.2

        fig, ax = plt.subplots(figsize=(max(6, len(common_domains) * 2), 4))

        ax.bar(x - 1.5 * bar_width, old_sr["overall_success_rate"], width=bar_width,
               color=old_color, alpha=0.45, label="Old (initial SR)")
        ax.bar(x - 0.5 * bar_width, old_sr["improved_success_rate"], width=bar_width,
               color=old_color, alpha=1.0, label="Old (improved SR)")
        ax.bar(x + 0.5 * bar_width, new_sr["overall_success_rate"], width=bar_width,
               color=new_color, alpha=0.45, label="New (initial SR)")
        ax.bar(x + 1.5 * bar_width, new_sr["improved_success_rate"], width=bar_width,
               color=new_color, alpha=1.0, label="New (improved SR)")

        ax.set_xticks(x)
        ax.set_xticklabels(common_domains)
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rates by Domain: Old vs New (Initial vs Improved)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        if self.figures_dir:
            plt.savefig(f"{self.figures_dir}/success_rates_comparison_by_domain_old_vs_new.pdf")
        plt.show()

    # Validity improvement per domain / task_name / model_name
    def check_validity_improvement(self, df, group_by, validity_col, improved_col):
        grouped = df.groupby(group_by).agg(overall_success_rate=(validity_col, "mean"), improved_success_rate=(improved_col, "mean"))
        grouped["improvement_pct"] = ((grouped["improved_success_rate"] - grouped["overall_success_rate"]) / grouped["overall_success_rate"] * 100)
        print(f"=== Validity improvement by {group_by} for {'old' if df is self.df_old else 'new'} results ===")
        print(grouped.round(2))

        return grouped

    def compare_validity_improvement(self, group_bys=("domain", "task_name"), validity_col="curr_validity", improved_col="cdt_subseq_validity"):
        results = {}
        for group_by in group_bys:
            old_g = self.check_validity_improvement(self.df_old, group_by, validity_col, improved_col)
            new_g = self.check_validity_improvement(self.df_new, group_by, validity_col, improved_col)

            combined = old_g.add_suffix("_old").join(new_g.add_suffix("_new"), how="inner")
            combined["improvement_pct_change"] = (combined["improvement_pct_new"] - combined["improvement_pct_old"])
            combined = combined.round(2)
            results[group_by] = combined

            print(f"=== Validity improvement by {group_by.replace("_", " ")} (old vs new) ===")
            print(combined.to_string())

            if self.tables_dir:
                latex_path = f"{self.tables_dir}/validity_improvement_{group_by}.tex"
                combined.to_latex(latex_path, float_format="%.2f")
                print(f"Saved LaTeX table to {latex_path}")

            fig, ax = plt.subplots(figsize=(max(6, len(combined) * 0.6), 4))
            combined[["improvement_pct_old", "improvement_pct_new"]].plot(kind="bar", ax=ax)
            ax.set_ylabel("Improvement in Success Rate (%)")
            ax.set_title(f"Validity Improvement by {group_by.replace("_", " ")}: Old vs New")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            if self.figures_dir:
                plt.savefig(f"{self.figures_dir}/validity_improvement_{group_by}.pdf")
            plt.show()

        return results

    # Success rates at lengths, per domain / per model
    def compare_success_rates_at_lengths(self, validity_column_1="curr_validity",
                                         validity_column_2="cdt_subseq_validity",
                                         plan_column="curr_plan", group_by="optimal", split_by=None):
        """
        split_by: "domain", "model_name", "model_family", etc., or None for the
        overall (ungrouped) results. Runs extract_success_rates_at_lengths separately
        for each value, on both old and new results.
        """
        if split_by is None:
            values = [None]
        else:
            values = sorted(set(self.df_old[split_by].unique()) | set(self.df_new[split_by].unique()))

        results = {}
        for value in values:
            label = "Overall" if value is None else f"{split_by} = {value}"

            if value is None:
                old_subset = self.df_old
                new_subset = self.df_new
            else:
                old_subset = self.df_old[self.df_old[split_by] == value]
                new_subset = self.df_new[self.df_new[split_by] == value]

            print(f"\n--- {label} (OLD) ---")
            old_result = (
                self.old.extract_success_rates_at_lengths(
                    old_subset, validity_column_1, validity_column_2,
                    plan_column=plan_column, group_by=group_by, lengths=[2,4,6,8,10]
                ) if not old_subset.empty else None
            )
            if old_subset.empty:
                print(f"No old data for {label}")

            print(f"\n--- {label} (NEW) ---")
            new_result = (
                self.new.extract_success_rates_at_lengths(
                    new_subset, validity_column_1, validity_column_2,
                    plan_column=plan_column, group_by=group_by, lengths=[2,4,6,8,10]
                ) if not new_subset.empty else None
            )
            if new_subset.empty:
                print(f"No new data for {label}")

            results["Overall" if value is None else value] = {"old": old_result, "new": new_result}

        return results

    def compare_success_rates_at_lengths_graph(self, validity_column_1="curr_validity", validity_column_2="cdt_subseq_validity",
                                               plan_column="curr_plan", group_by="optimal", split_by="domain", mode="validity", save=True):
        """
        mode="validity":    plots both validity_column_1 and validity_column_2 as separate lines (linestyle), old vs new (color).
        mode="improvement": plots (validity_column_2 - validity_column_1) as a single line, old vs new (color). Easier to compare trends across models/domains.
        """
        values = sorted(set(self.df_old[split_by].unique()) | set(self.df_new[split_by].unique()))

        ncols = min(3, len(values))
        nrows = math.ceil(len(values) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axs_flat = axs.flatten()

        plan_length_label = "Optimal plan length" if group_by == "optimal" else "LLM-generated plan length"

        for ax, value in zip(axs_flat, values):
            old_subset = self.df_old[self.df_old[split_by] == value]
            new_subset = self.df_new[self.df_new[split_by] == value]

            old_result = (
                self.old.extract_success_rates_at_lengths(
                    old_subset, validity_column_1, validity_column_2,
                    plan_column=plan_column, group_by=group_by, verbose=False, lengths=[2,4,6,8,10]
                ) if not old_subset.empty else None
            )
            new_result = (
                self.new.extract_success_rates_at_lengths(
                    new_subset, validity_column_1, validity_column_2,
                    plan_column=plan_column, group_by=group_by, verbose=False, lengths=[2,4,6,8,10]
                ) if not new_subset.empty else None
            )

            if (old_result is None or old_result.empty) and (new_result is None or new_result.empty):
                ax.set_title(f"{value} (no data)")
                ax.axis("off")
                continue

            if mode == "improvement":
                for result, color, label in [(old_result, "tab:blue", "Old"), (new_result, "tab:orange", "New")]:
                    if result is not None and not result.empty:
                        improvement = result[validity_column_2] - result[validity_column_1]
                        ax.plot(result.index, improvement, marker="o", color=color, label=label)
                ax.axhline(0, color="gray", linestyle=":", linewidth=1)
                ax.set_ylabel(f"{validity_column_2} - {validity_column_1}")
            else:  # mode == "validity"
                for result, color, label in [(old_result, "tab:blue", "Old"), (new_result, "tab:orange", "New")]:
                    if result is not None and not result.empty:
                        ax.plot(result.index, result[validity_column_1], marker="o", linestyle="--",
                                color=color, label=f"{label} {validity_column_1}")
                        ax.plot(result.index, result[validity_column_2], marker="o", linestyle="-",
                                color=color, label=f"{label} {validity_column_2}")
                ax.set_ylabel("Success rate")

            ax.set_title(str(value))
            ax.set_xlabel(plan_length_label)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

        for ax in axs_flat[len(values):]:
            ax.axis("off")

        title_metric = f"improvement ({validity_column_2} - {validity_column_1})" if mode == "improvement" \
            else f"success rate ({validity_column_1} vs {validity_column_2})"
        fig.suptitle(f"{title_metric.capitalize()} by {plan_length_label.lower()}, grouped by {split_by}", fontsize=14)
        plt.tight_layout()
        if save and self.figures_dir:
            plt.savefig(f"{self.figures_dir}/success_rates_at_lengths_{split_by}_{mode}.pdf")
        plt.show()

    @staticmethod
    def get_model_family(model_name):
        name = str(model_name).lower()
        for prefix, family in MODEL_FAMILY_PREFIXES.items():
            if name.startswith(prefix):
                return family
        for keyword, family in MODEL_FAMILY_KEYWORDS.items():
            if keyword in name:
                return family
        return "other"

    def add_model_family_column(self):
        self.df_old["model_family"] = self.df_old["model_name"].apply(self.get_model_family)
        self.df_new["model_family"] = self.df_new["model_name"].apply(self.get_model_family)

    def validity_improvement_standalone(self, group_by="model_name",
                                        validity_col="curr_validity", improved_col="cdt_subseq_validity"):
        """
        Like _validity_improvement, but for old and new separately with no join —
        use this when the group values (e.g. model names) aren't comparable 1:1
        across the two result sets.
        """
        old_g = self.check_validity_improvement(self.df_old, group_by, validity_col, improved_col).round(2)
        new_g = self.check_validity_improvement(self.df_new, group_by, validity_col, improved_col).round(2)

        print(f"=== Validity improvement by {group_by} for OLD results ===")
        print(old_g.to_string())
        print(f"\n=== Validity improvement by {group_by} for NEW results ===")
        print(new_g.to_string())

        if self.tables_dir:
            old_path = f"{self.tables_dir}/validity_improvement_{group_by}_old.tex"
            new_path = f"{self.tables_dir}/validity_improvement_{group_by}_new.tex"
            old_g.to_latex(old_path, float_format="%.2f")
            new_g.to_latex(new_path, float_format="%.2f")
            print(f"\nSaved LaTeX tables to {old_path} and {new_path}")

        fig, axs = plt.subplots(1, 2, figsize=(max(10, (len(old_g) + len(new_g)) * 0.5), 4), sharey=True)
        old_g["improvement_pct"].plot(kind="bar", ax=axs[0], color="tab:blue")
        axs[0].set_title(f"Old: Improvement by {group_by.replace("_", " ")}")
        axs[0].set_ylabel("Improvement in Success Rate (%)")
        axs[0].grid(True, axis="y", linestyle="--", alpha=0.5)

        new_g["improvement_pct"].plot(kind="bar", ax=axs[1], color="tab:orange")
        axs[1].set_title(f"New: Improvement by {group_by.replace("_", " ")}")
        axs[1].grid(True, axis="y", linestyle="--", alpha=0.5)

        for ax in axs:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        if self.figures_dir:
            plt.savefig(f"{self.figures_dir}/validity_improvement_{group_by}_standalone.pdf")
        plt.show()

        return old_g, new_g

    def get_union_results(self, tag_source=True):
        """
        Concatenates old and new results into a single DataFrame.
        Columns present in only one side become NaN on the other, via outer concat.
        """
        old = self.df_old.copy()
        new = self.df_new.copy()

        if tag_source:
            old["results_source"] = "old"
            new["results_source"] = "new"

        union_df = pd.concat([old, new], ignore_index=True, sort=False)
        return union_df

    def run_union_analysis(self, union_dir, success_level=None, domain=None, run_analysis=True, figures_dir=None, tables_dir=None):
        """
        Writes the union of old + new results to its own postprocessed CSV and
        loads it as a full AnalyseResults instance (plots/analysis included,
        unless run_analysis=False).
        """
        os.makedirs(union_dir, exist_ok=True)
        union_df = self.get_union_results()

        union_file = f"{union_dir}/full_evaluation_union_postprocessed.csv"
        union_df.to_csv(union_file, index=False)

        self.union = AnalyseResults(union_file, union_dir, is_processed=True, success_level=success_level, domain=domain, run_analysis=run_analysis, figures_dir=figures_dir, tables_dir=tables_dir)
        self.df_union = self.union.results
        return self.union

def resolve_results_file(project_dir, date_suffix, domain, missing_instances=False):
    """
    Mirrors the file-resolution logic from main(): finds the (post-processed,
    if available) results file for a given run, identified by its date suffix.
    """
    results_dir = f"{project_dir}/results/plan_recovery/final_evaluation{date_suffix}/"

    if domain:
        results_file = f"{results_dir}/full_evaluation_{domain}.csv"
    else:
        results_file = f"{results_dir}/full_evaluation.csv"

    if missing_instances:
        results_file = results_file.replace(
            "full_evaluation", "full_evaluation_w_missing_instances"
        )

    post_processed = results_file.replace(".csv", "_postprocessed.csv")
    is_processed = False
    if os.path.exists(post_processed):
        results_file = post_processed
        is_processed = True

    return results_file, results_dir, is_processed


if __name__ == "__main__":
    current_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    figures_dir = os.path.join(project_dir, "figures/plan_recovery/comparison_old_vs_new")
    tables_dir = os.path.join(project_dir, "tables/plan_recovery/comparison_old_vs_new")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    comparer = CompareResults(project_dir, domain=None, success_level=None, missing_instances=False, tables_dir=tables_dir, figures_dir=figures_dir)

    # figures_union_dir = f"{project_dir}/figures/plan_recovery/final_evaluation_union/"
    # tables_union_dir = f"{project_dir}/tables/plan_recovery/final_evaluation_union/"
    # union_dir = f"{project_dir}/results/plan_recovery/final_evaluation_union/"
    #
    # os.makedirs(figures_union_dir, exist_ok=True)
    # os.makedirs(tables_union_dir, exist_ok=True)
    # os.makedirs(union_dir, exist_ok=True)

    # comparer.run_union_analysis(union_dir, success_level=None, domain=None, run_analysis=True, figures_dir=figures_union_dir, tables_dir=tables_union_dir)