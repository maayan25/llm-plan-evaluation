# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to generate all solutions using SymK planner, and validating them with VAL.
import os
import json
import tempfile
from pathlib import Path
from utils import get_config_dir, get_instances_dir

from tarski.io import PDDLReader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
parent_dir = os.path.abspath(os.path.join(project_dir, os.pardir))

# TODO add symk to README setup instructions
llm_project_dir = os.path.join(parent_dir, "LLM-Planning-PlanBench/")

# Function from llm_planning_analysis/prompt_generation.py
def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)

class GenerateGTPlans:
    def __init__(self, config, validate=False, generate=True, max_instances=None):
        self.config_name = config
        self.domain_dir = get_config_dir(self.config_name)
        self.domain_file =  f"{self.domain_dir}/generated_domain.pddl"
        self.instances = self.get_instances_from_config(max_instances) # {instance_number: int, path_to_file: str}
        self.SYMK_PATH = os.path.join(parent_dir, os.pardir, "planners/symk")
        self.VAL_PATH = os.path.join(llm_project_dir, "planner_tools/VAL")
        self.generated_plans = None
        self.valid_plans = None

        # Directory to store solutions
        self.output_dir = f"{project_dir}/results/gt_plans/{config}"
        os.makedirs(self.output_dir, exist_ok=True)

        # TODO maybe adapt this according to the astar GT plan?
        self.cost_threshold = 20 # the max cost for which a plan is searched

        output_filename = "gt_plans.json"
        if generate:
            # Check if there are some generated plans already in the output directory, if so, load them and skip generation
            if os.path.exists(f"{self.output_dir}/{output_filename}"):
                with open(f"{self.output_dir}/{output_filename}", "r") as file:
                    data = json.load(file)
                    self.generated_plans = data["instances"]
                    if max_instances is not None:
                        # Filter the loaded plans to only include those with instance numbers less than max_instances
                        self.generated_plans = {k: v for k, v in self.generated_plans.items() if int(k) < max_instances}
                    # Convert keys back to integers if they were saved as strings
                    self.generated_plans = {int(k): v for k, v in self.generated_plans.items()}
                print(f"Loaded previously generated plans for {len(self.generated_plans)} instances from {self.output_dir}/{output_filename}")
            print("Generating GT plans...")
            self.generated_plans = self.generate_plans()
            # self.save_plans_to_json(self.generated_plans, filename=output_filename)
            print(f"Generation complete. Saving plans for {len(self.generated_plans)} instances to {self.output_dir}/{output_filename}")
        else:
            # Load previously generated plans from JSON file
            with open(f"{self.output_dir}/{output_filename}", "r") as file:
                data = json.load(file)
                self.generated_plans = data["instances"]
                if max_instances is not None:
                    # Filter the loaded plans to only include those with instance numbers less than max_instances
                    self.generated_plans = {k: v for k, v in self.generated_plans.items() if int(k) < max_instances}
                # Convert keys back to integers if they were saved as strings
                self.generated_plans = {int(k): v for k, v in self.generated_plans.items()}
            print(f"Loaded previously generated plans for {len(self.generated_plans)} instances from {self.output_dir}/{output_filename}")

        if validate:
            print("Validating GT plans...")
            self.valid_plans = self.validate_plans()
            # self.save_plans_to_json(self.valid_plans, filename="valid_plans.json")
            print(f"Validation complete. Saving valid plans for {len(self.valid_plans)} instances to {self.output_dir}/valid_plans.json")

    def get_instances_from_config(self, max_instances=None):
        """
        Get paths to instance files from their directory and add them to the instances dictionary with their instance number
        :return: a dictionary of instance number and the path to that instance's file
        """
        instances_dir = get_instances_dir(self.config_name)
        instances = {}
        path_to_instances = os.path.join(self.domain_dir, instances_dir)
        # If max_instances is specified, only consider instance files with numbers less than max_instances
        for file in os.listdir(path_to_instances):
            if "amended" in file or not file.endswith(".pddl"):
                # print(f"Skipping file: {file}")
                continue
            elif max_instances is not None and int(file.split("-")[1].split(".")[0]) > max_instances:
                # print(f"Skipping file: {file} as it exceeds max_instances limit of {max_instances}")
                continue
            # print(f"Found instance file: {file}")
            inst_num = int(file.split("-")[1].split(".")[0])  # instance-<number>.pddl
            instances[inst_num] = os.path.join(path_to_instances, file)
            # i += 1
        print(f"Got {len(instances)} instances from {instances_dir}")
        return instances

    def generate_plans(self):
        generated_plans = {}
        # Check if self.generated_plans already has plans for some instances, if so, skip those instances
        if self.generated_plans is not None:
            for inst_num in self.generated_plans.keys():
                if inst_num in self.instances:
                    # print(f"Skipping instance {inst_num} as it already has generated plans.")
                    del self.instances[inst_num]
        for inst_num, instance in self.instances.items():
            print(f"Generating plans for instance number: {inst_num}")
            # problem = get_problem(instance, self.domain_file)
            plan_files, plans = self.compute_plans_for_instance(instance, inst_num)
            generated_plans[inst_num] = plans
            self.append_plans_to_json(inst_num, plans, filename="gt_plans.json")

        return generated_plans

    def compute_plans_for_instance(self, problem, inst_num):
        """Run SYMK on all problems."""
        top_k = 5
        assert os.path.exists(f"{self.SYMK_PATH}/fast-downward.py")
        print(f"Running SYMK with problem: {problem}")
        cmd = f"{self.SYMK_PATH}/fast-downward.py {self.domain_file} {problem} --search \"symk_bd(plan_selection=top_k(num_plans={top_k}))\" > /dev/null"
        os.system(cmd)

        plans = self.extract_plans(inst_num, top_k)
        plan_files = [f"sas_plan.{i}" for i in range(1, len(plans) + 1)]
        print(f"Generated {len(plan_files)} Plans for instance {inst_num}")
        return plan_files, plans

    def extract_plans(self, inst_num, top_k):
        """Extract and save plans from Fast Downward output."""
        plans = []
        for plan_index in range(1, top_k + 1):
            plan_file = f"sas_plan.{plan_index}"
            try:
                with open(plan_file) as f:
                    plan = [line.rstrip() for line in f][:-1]
                    readable_plan = '\n'.join(plan)
                    # print(f"Generated plan for index {plan_index}: {readable_plan.split('\n')}")
                # os.remove(plan_file)
                plans.append(readable_plan)
            except FileNotFoundError:
                print(f"No plan found for instance {inst_num} at index {plan_index}. Stopping plan extraction.")
                break

        return plans

    def validate_plans(self):
        """
        For each instance check which plans are valid and save these
        :return:
        """
        valid_plans = {}
        for inst_num, instance in self.instances.items():
            plans = self.generated_plans[inst_num]
            print(f"Validating {len(plans)} plans for instance number: {inst_num}")
            # Validate plans for the instance using VAL, return a list of booleans indicating validity for each plan
            inst_plan_validity = self.validate_plans_for_instance(self.domain_file, instance, plans)

            # Rewrite the plans dictionary to remove invalid plans
            for plan, is_valid in zip(plans, inst_plan_validity):
                if is_valid:
                    if inst_num not in valid_plans:
                        valid_plans[inst_num] = []
                    valid_plans[inst_num].append(plan)

            self.append_plans_to_json(inst_num, valid_plans[inst_num], filename="valid_plans.json")

        return valid_plans

    def validate_plans_for_instance(self, domain, problem, plans):
        """Validate plans using VAL for a given domain and problem instance."""
        valid_plans = []

        with tempfile.TemporaryDirectory() as tmpdir:
            plan_paths = []
            for i, plan in enumerate(plans):
                p = Path(tmpdir) / f"{i}.txt"
                p.write_text(plan)
                plan_paths.append(str(p))

            cmd = f"{self.VAL_PATH}/validate {domain} {problem} {' '.join(plan_paths)} 2>&1 | tee plan_validation_output.log"
            response = os.popen(cmd).read()

            # print(f"PREAMBLE: {response.split("Checking plan: ")[0]}")
            chunks = response.split("Checking plan: ")[1:]  # first split part is preamble (empty...)

            for plan_file, chunk in zip(plan_paths, chunks):
                if "Plan valid" in chunk:
                    # print(f"Plan {plan_file} is valid for instance {problem}")
                    valid_plans.append(True)
                else:
                    print(f"Plan {plan_file} is invalid for instance {problem}. Response from VAL: {chunk}")
                    print(f"Content of {plan_file} \n{Path(plan_file).read_text()}")
                    valid_plans.append(False)

        return valid_plans

    def save_plans_to_json(self, plans, filename="gt_plans.json"):
        """Save valid plans to a JSON file"""
        output_file = f"{self.output_dir}/{filename}"
        plans = {str(k): v for k, v in plans.items()}
        data = {"config": self.config_name, "instances": plans}

        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                existing_data = json.load(file)
            overwritten = set(existing_data["instances"]) & set(data["instances"])
            if overwritten:
                print(f"Overwriting {len(overwritten)} existing instance(s): {sorted(overwritten)}")
            existing_data["instances"].update(data["instances"])
            data = existing_data

        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)

    def append_plans_to_json(self, instance_number, plans, filename="gt_plans.json"):
        """Add a plans for a single instance to the JSON file for a specific instance number."""
        output_file = f"{self.output_dir}/{filename}"
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                data = json.load(file)
        else:
            data = {"config": self.config_name, "instances": {}}

        # Convert instance_number to string for JSON compatibility
        instance_key = str(instance_number)
        if instance_key not in data["instances"]:
            data["instances"][instance_key] = []

        data["instances"][instance_key] = plans

        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)


def generate_gt_plans(domain, validate=False, generate=True, max_instances=None):
    """
    Generate and validate GT plans for a given domain using the GenerateGTPlans class.
    :param domain: the domain name (e.g., "blocksworld", "logistics") for which to generate and validate plans
    :param validate: whether to validate the generated plans using VAL (default: False)
    :param generate: whether to generate new plans or load existing ones from a JSON file (default: True)
    :return:
    """
    plan_generator = GenerateGTPlans(domain, validate=validate, generate=generate, max_instances=max_instances)
    return plan_generator.generated_plans, plan_generator.valid_plans

if __name__ == "__main__":
    config = "elevator"
    generator = GenerateGTPlans(config, validate=True, generate=True)
