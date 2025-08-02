from evaluate_plan import Plan
import os
import pandas as pd
import ast

def update_plan_info_in_csv(input_dir):
    """
    For each CSV file in the directory:
    - Parses columns to Python objects.
    - Constructs Plan objects for each plan type.
    - Computes search_steps.
    - Adds new step columns.
    - Saves to new CSV with '_fixed' suffix.
    """

    plan_types = [
        ("curr_plan", "curr_validity", "curr_steps"),
        ("cdt_plan", "cdt_validity", "cdt_steps"),
        ("subseq_plan", "subseq_validity", "subseq_steps"),
        ("cdt_subseq_plan", "cdt_subseq_validity", "cdt_subseq_steps"),
    ]

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv") and not filename.startswith("full_evaluation") and not filename.endswith("_fixed.csv"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")

            df = pd.read_csv(file_path)
            for col in ["curr_plan", "cdt_plan", "subseq_plan", "cdt_subseq_plan", "gt_plan"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            original_df = df.copy(deep=True)
            df = update_stv_in_df(df, plan_types)
            for col in df.columns:
                if col not in ["curr_steps", "cdt_steps", "subseq_steps", "cdt_subseq_steps"]:
                    if not df[col].equals(original_df[col]):
                        print(f"⚠️ Column changed unexpectedly: {col}")

            # Save to new file
            # new_filename = filename.replace(".csv", "_fixed.csv")
            df.to_csv(os.path.join(input_dir, filename), index=False)
            print(f"Saved fixed file as {filename}")

def update_stv_in_df(df: pd.DataFrame, plan_types: list) -> pd.DataFrame:
    """
    Create 4 Plan objects from a DataFrame row for each plan type and compute the steps to validity.
    """
    # Process each row
    for idx, row in df.iterrows():
        for plan_col, valid_col, steps_col in plan_types:
            try:
                plan = Plan(
                    plan=row[plan_col],
                    gt=row["gt_plan"],
                    valid=row[valid_col],
                    domain=row["domain"],
                    instance_id=row["instance_id"],
                    to_simulate=False,
                    unique_id=f"{row['model']}_{row['task']}_{row['domain']}_{row['instance_id']}"
                )
                steps = search_steps_to_validity(plan)
                print(f"Plan {row[plan_col]} got {len(steps)} steps to validity ({steps})")
                df.at[idx, steps_col] = len(steps)
            except Exception as e:
                print(f"Error in row {idx}, {plan_col}: {e}")
                df.at[idx, steps_col] = None
    return df

def search_steps_to_validity(plan: Plan) -> list:
    """
    Search for the steps to validity in the plan; the steps to validity are the steps that are needed to get from
    the current plan to the ground truth plan.
    This is based on actions, not on the states:
        - adding actions to it
        - removing actions from it
        - changing the order of actions in it
        - changing the variables of actions in it
    :return: the steps to validity as a dictionary of what to do and the action to do it
    """
    # TODO take care of repeated actions; also if have time, check where variables need swapping
    # TODO need to check if the action is repaired, if it needs reordering

    steps_to_validity = []
    better_plan_lst = []
    no_of_repairs = 0

    improved = False

    # Remove the auxiliary actions from the plan
    plan_list = plan.get_plan()
    for i in range(len(plan_list)):
        try:
            action = plan_list[i]
            if plan.plan_trace[i] != "redundant":
                # remove the auxiliary actions from the plan trace
                better_plan_lst.append(action)
            else:
                steps_to_validity.append(("remove_action", action))
                improved = True
        except KeyError as e:
            print(f"KeyError: {i} for plan {plan_list} and plan trace {plan.plan_trace}")

    if improved:
        # Create a new plan with these actions, to check the steps to validity (positioning might change now)
        better_plan = Plan(better_plan_lst, plan.gt_plan, False, plan.config_name, plan.instance_id, unique_id=f"{plan.unique_id}_better")
    else:
        better_plan = plan

    if better_plan.is_valid:
        # Plan is valid and no steps to validity needed for plan (except for the redundant actions if they exist)
        plan.set_steps_to_validity(steps_to_validity)
        return steps_to_validity

    plan_trace = better_plan.plan_trace
    assert len(better_plan_lst) <= len(plan.gt_plan)  # only action pairs should remain (actions from GT could be missing)

    # On the new plan trace, add the fixes + reorder for every action which is not in the right position
    for i in range(len(better_plan_lst)):
        action = better_plan_lst[i]
        if action in plan.gt_plan:
            if plan_trace[i] == "position":
                continue
            else:  # plan_trace[i] == "correct"
                steps_to_validity.append(("reorder_action", action))
        else:
            if plan_trace[i] == "same_act":
                # If the action is in the GT plan, but in the wrong position, it needs to be moved
                no_of_repairs += 1
                steps_to_validity.append(("fix_variables", action))
            elif plan_trace[i] == "diff_act":
                # If the action has little similarity to the GT action, it needs to be fixed
                no_of_repairs += 1
                steps_to_validity.append(("fix_action", action))
            else: # plan_trace[i] == "auxiliary"
                # If the action is not in the GT plan, it needs to be removed
                steps_to_validity.append(("remove_action", action))

    # TODO actually check if self.action_pairs if this action has a pair, if not, add it to the steps to validity
    for action in plan.gt_plan:
        if action not in plan_list:
            if no_of_repairs > 0:
                no_of_repairs -= 1
            else:
                # The action is not in the plan, it needs to be added
                steps_to_validity.append(("add_action", action))

    plan.set_steps_to_validity(steps_to_validity)
    return steps_to_validity

if __name__ == "__main__":
    current_dir = os.getcwd()
    project_dir = os.path.join(current_dir, os.pardir, os.pardir)
    input_dir = os.path.join(project_dir, "results/plan_recovery/final_evaluation/o4-mini")
    update_plan_info_in_csv(input_dir)
    print("All CSV files processed and fixed.")