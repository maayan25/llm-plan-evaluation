import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
results_dir = f"{project_dir}/results/plan_recovery/final_evaluation_param_search_first_round"
INSTANCE_COL = "instance_id"
TIME_COL = "search_time"
EVALS_COL = "evals"
LEN_COL = None
PARAMS = ["top_k", "min_k", "max_k", "tol", "refine_iters"]
DEFAULTS = dict(top_k=10, min_k=7, max_k=25, tol=0.1, refine_iters=2)


def to_bool(s):
    return s.astype(str).str.strip().str.lower().eq("true")


def parse_file(f: Path):
    """results/{domain}/{model}/evaluation_{task}_{params}.csv -> domain, model, task, params"""
    domain, model = f.parts[-3], f.parts[-2]
    stem = f.stem
    params = dict(DEFAULTS)
    for p in PARAMS:
        m = re.search(rf"{p}[_=]?(\d+(?:\.\d+)?)", stem)
        if m:
            params[p] = float(m.group(1)) if p == "tol" else int(float(m.group(1)))
    m = re.match(r"evaluation_(.*?)_?(?:top_k|min_k|max_k|tol|refine_iters)", stem)
    task = m.group(1) if m else stem.replace("evaluation_", "")
    return domain, model, task, params


def load():
    frames = []
    for f in sorted(Path(results_dir).glob("*/*/evaluation_*.csv")):
        df = pd.read_csv(f)
        domain, model, task, params = parse_file(f)
        df["domain_"], df["model_"], df["task_"] = domain, model, task
        for p, v in params.items():
            df[f"p_{p}"] = v
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["curr"] = to_bool(df["curr_validity"])
    df["cdt"] = to_bool(df["cdt_subseq_validity"])
    df["rescued"] = df["cdt"] & ~df["curr"]
    df["cfg"] = df.apply(lambda r: "_".join(f"{p}{r[f'p_{p}']:g}" for p in PARAMS), axis=1)
    return df


def summarise(df, by_cols):
    """Aggregate per config (and optionally extra columns). Timeouts = instances missing from a config."""
    grp = ["domain_", "model_", "task_"]
    universe = df.groupby(grp)[INSTANCE_COL].nunique().rename("expected").reset_index()

    agg = {"n": ("cdt", "size"), "rescued": ("rescued", "sum"), "final_valid": ("cdt", "sum"),
           "mean_time": (TIME_COL, "mean"), "total_time": (TIME_COL, "sum")}
    if EVALS_COL in df.columns:
        agg["mean_evals"] = (EVALS_COL, "mean")

    keys = list(dict.fromkeys(["cfg"] + grp + by_cols))   # dedupe, keeps order
    per = df.groupby(keys).agg(**agg).reset_index()
    per = per.merge(universe, on=grp)

    out = per.groupby(["cfg"] + by_cols).agg(
        n=("n", "sum"), expected=("expected", "sum"), rescued=("rescued", "sum"),
        final_valid=("final_valid", "sum"), total_time=("total_time", "sum"),
        **({"mean_evals": ("mean_evals", "mean")} if EVALS_COL in df.columns else {}),
    ).reset_index()
    out["rescue_rate"] = out["rescued"] / out["expected"]
    out["frac_of_valid_rescued"] = out["rescued"] / out["final_valid"].replace(0, np.nan)
    out["timeouts"] = out["expected"] - out["n"]
    out["mean_time"] = out["total_time"] / out["n"]
    return out

def pareto(t):
    """Configs where no other config is both faster and better."""
    keep, best = [], -np.inf
    for i, r in t.sort_values(["mean_time", "rescue_rate"], ascending=[True, False]).iterrows():
        if r["rescue_rate"] > best:
            keep.append(i)
            best = r["rescue_rate"]
    return t.loc[keep]


def ofat_tables(df, table):
    """Effect of each parameter with all others at their defaults."""
    cfgs = df.drop_duplicates("cfg")[["cfg"] + [f"p_{p}" for p in PARAMS]].set_index("cfg")
    for p in PARAMS:
        others = [q for q in PARAMS if q != p]
        mask = np.logical_and.reduce([cfgs[f"p_{q}"] == DEFAULTS[q] for q in others])
        sub = table.set_index("cfg").loc[cfgs[mask].index]
        sub = sub.assign(**{p: cfgs[mask][f"p_{p}"]}).sort_values(p)
        if len(sub) > 1:
            print(f"\n--- {p} (others at default) ---")
            print(sub[[p, "rescue_rate", "frac_of_valid_rescued", "timeouts", "mean_time"]]
                  .round(3).to_string(index=False))


def main():
    df = load()
    table = summarise(df, [])
    cols = ["cfg", "rescue_rate", "frac_of_valid_rescued", "rescued", "timeouts", "mean_time"]
    if EVALS_COL in df.columns:
        cols.append("mean_evals")
    print(table.sort_values("rescue_rate", ascending=False)[cols].round(3).to_string(index=False))

    pf = pareto(table)
    print(pf[cols].round(3).to_string(index=False))

    ofat_tables(df, table)

    # Per domain/model breakdown
    for by in (["domain_"], ["model_"]):
        t = summarise(df, by)
        print(f"\n=== By {by[0]} (rescue rate, configs as rows) ===")
        print(t.pivot(index="cfg", columns=by[0], values="rescue_rate").round(3).to_string())

    if LEN_COL and LEN_COL in df.columns:
        df["len_bin"] = pd.qcut(df[LEN_COL], 3, labels=["short", "medium", "long"], duplicates="drop")
        t = summarise(df, ["len_bin"])
        print("\n=== By plan length (rescue rate) ===")
        print(t.pivot(index="cfg", columns="len_bin", values="rescue_rate").round(3).to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(table["mean_time"], table["rescue_rate"], alpha=0.6)
    ax.plot(pf["mean_time"], pf["rescue_rate"], "r-o", label="Pareto frontier")
    for _, r in pf.iterrows():
        ax.annotate(r["cfg"], (r["mean_time"], r["rescue_rate"]), fontsize=6, rotation=20)
    ax.set_xlabel("Mean search time per instance (s)")
    ax.set_ylabel("Rescue rate (cdt valid & curr invalid) / all instances")
    ax.legend()
    fig.tight_layout()
    fig.savefig("param_search_pareto.png", dpi=200)
    table.to_csv("param_search_summary.csv", index=False)


if __name__ == "__main__":
    main()