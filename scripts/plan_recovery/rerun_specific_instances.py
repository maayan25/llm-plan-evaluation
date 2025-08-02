# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# This script reruns specific instances of evaluation tasks based on a CSV file.

import os
import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# dict_lock = threading.Lock()

def run_instance(row):
    domain = row['domain']
    task = row['task']
    if task == "_one_shot":
        task = ""
    model = row['model']
    instance_id = row['instance_id']

    cmd = ["python", "run_evaluation.py", "--domain", str(domain), "--task", str(task), "--model", str(model), "--specific_instance", str(instance_id)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Completed instance {instance_id} for {domain}, {task}, {model}")
        return f"Success: {domain}, {task}, {model}, {instance_id}"
    except subprocess.CalledProcessError as e:
        print(f"Error running instance {instance_id} for {domain}, {task}, {model}: {e.stderr}")
        return f"Error ({domain}, {task}, {model}, {instance_id}):\n{e.stderr}"

def main(csv_path, max_workers):
    df = pd.read_csv(csv_path)
    i = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_instance, row) for _, row in df.iterrows()]
        # futures = []
        # for _, row in df.iterrows():
        #     print(f"Submitting instance {i}/{len(df)}: {row['domain']}, {row['task']}, {row['model']}, {row['instance_id']}")
        #     if i < 5:
        #         futures.append(executor.submit(run_instance, row))
        #     else:
        #         print("Skipping further instances for demonstration purposes.")
        #         break

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file from the project root directory")
    parser.add_argument("--workers", type=int, default=20, help="Max number of parallel workers")
    args = parser.parse_args()

    current_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    path_to_csv = os.path.join(project_dir, args.csv)

    main(path_to_csv, args.workers)
