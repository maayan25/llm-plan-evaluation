import pandas as pd
from nltk.corpus import wordnet as wn
import os
import re

def extract_action_names_from_domain(domain_file):
    """
    Extracts action names from a PDDL domain file.
    Handles multi-word action names like 'pick-up' or 'stack blocks'.
    """
    actions = []
    with open(domain_file, "r") as f:
        content = f.read().lower()

    # Regex to match (:action <name>)
    matches = re.findall(r"\(:action\s+([^\s\)]+)", content)
    for match in matches:
        action_name = match.replace("_", " ").replace("-", " ")
        actions.append(action_name.strip())
    return actions


def check_wordnet_coverage(action_names):
    """
    Check WordNet coverage for a list of action names.
    Returns a DataFrame with coverage information.
    """
    results = []
    for act in action_names:
        # act_norm = act.lower()
        act_norm = act.lower().replace("_", " ").replace("-", " ")

        syns = wn.synsets(act_norm, pos=wn.VERB) + wn.synsets(act_norm, pos=wn.NOUN)
        covered = len(syns) > 0
        if not covered:
            for word in act_norm.split():
                if wn.synsets(word, pos=wn.VERB) or wn.synsets(word, pos=wn.NOUN):
                    covered = True
                    break

        results.append({
            "action": act,
            "covered": covered
        })

    df = pd.DataFrame(results)
    return df

def check_domain_directory_coverage(file_paths):
    """
    Extracts action names from all domain files in a directory and
    checks their WordNet coverage.
    """
    all_actions = []
    for file in file_paths:
        if file.endswith(".pddl"):
            actions = extract_action_names_from_domain(file)
            all_actions.extend(actions)

    all_actions = sorted(set(all_actions))
    return check_wordnet_coverage(all_actions)

if __name__ == "__main__":
    current_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    results_dir = f"{project_dir}/results/plan_recovery/final_evaluation/"
    results_file = f"{results_dir}/full_evaluation.csv"

    is_IPC = True
    domain_paths = []
    if is_IPC:
        all_domains = {
            "ipc-1998": ["assembly-round-1-adl", "grid-round-2-strips", "gripper-round-1-strips", "movie-round-1-strips"],
            "ipc-2000": ["blocks-strips-typed", "elevator-strips-simple-typed", "freecell-strips-typed",
                            "logistics-strips-typed"],
            "ipc-2002": ["depots-strips-automatic", "driverlog-strips-automatic", "rovers-strips-automatic",
                            "satellite-strips-automatic", "zenotravel-strips-automatic"],
            "ipc-2004": ["airport-nontemporal-strips", "pipesworld-no-tankage-nontemporal-strips",
                        "promela-dining-philosophers-strips", "psr-small-strips", "settlers-strips",
                        "umts-flaw-temporal-strips"],
            "ipc-2006": ["openstacks-metric-time-strips", "pathways-propositional-strips",
                         "storage-preferences-simple", "tpp-propositional-strips", "trucks-propositional-strips"],
            "ipc-2008": ["crew-planning-temporal-satisficing-strips", "cyber-security-sequential-satisficing-strips",
                         "model-train-temporal-satisficing-numeric-fluents", "parc-printer-sequential-satisficing-strips",
                         "peg-solitaire-sequential-satisficing-strips", "scanalyzer-3d-sequential-satisficing-strips",
                         "sokoban-sequential-satisficing-strips", "transport-sequential-satisficing-strips",
                         "woodworking-sequential-optimal-strips"],
            "ipc-2011": ["barman-sequential-optimal", "floor-tile-sequential-satisficing", "match-cellar-temporal-satisficing",
                         "no-mystery-sequential-satisficing", "parking-sequential-satisficing", "tidybot-sequential-satisficing",
                         "transport-sequential-satisficing", "visit-all-sequential-satisficing"],
            "ipc-2014": ["cave-diving-sequential-optimal", "child-snack-sequential-satisficing", "city-car-sequential-satisficing",
                         "floor-tile-sequential-satisficing", "genome-edit-distances-sequential-satisficing", "hiking-sequential-satisficing",
                         "maintenance-sequential-satisficing", "map-analyzer-temporal-satisficing", "road-traffic-accident-management-temporal-satisficing",
                         "tetris-sequential-satisficing", "thoughtful-sequential-satisficing", "turn-and-open-temporal-satisficing"]
        }
        # Coverage: 96.17% (176 / 183) for IPC domains 2000-2004
        # Total coverage: 99.99% (97647 / 97657) for all IPC domains 1998-2014
        config_dir = os.path.join(project_dir, os.pardir, "kg_modelling", "pddl-instances")
        for year, domains in all_domains.items():
            for domain in domains:
                domain_name = "domain.pddl"
                other = ["airport", "promela", "psr", "openstacks", "pathways", "tpp", "trucks", "cyber-security",
                         "parc-printer"]
                for o in other:
                    if o in domain:
                        domain_name = "domains/domain-1.pddl"
                domain_path = os.path.join(config_dir, year, "domains", domain, domain_name)
                domain_paths.append(domain_path)
    else:
        domains = ["blocksworld", "logistics", "office_robot"]
        # BW 75%, logistics 100%, office_robot 100% = Total 96.15%
        config_dir = os.path.join(project_dir, os.pardir, "LLM-Planning-PlanBench", "llm_planning_analysis", "instances")
        for domain in domains:
            domain_path = os.path.join(config_dir, domain, f"generated_domain.pddl")
            domain_paths.append(domain_path)

    df_cov = check_domain_directory_coverage(domain_paths)
    print(df_cov)

    print("\n" + "="*50 + "\n")

    # actions = ["remove", "unload", "deconstruct", "unstack"]
    # df_cov = check_wordnet_coverage(actions)
    # print(df_cov)

    # # Similarity from actions list to "stack" to get the highest similarity
    # target = "stack"
    # for act in actions:
    #     syn1 = wn.synsets(target, pos=wn.VERB) + wn.synsets(target, pos=wn.NOUN)
    #     syn2 = wn.synsets(act, pos=wn.VERB) + wn.synsets(act, pos=wn.NOUN)
    #     sim = 0.0
    #     if syn1 and syn2:
    #         sim = syn1[0].wup_similarity(syn2[0])
    #     print(f"Similarity between '{target}' and '{act}': {sim}")

    # Similarity between 'stack' and 'remove': 0.2857142857142857
    # Similarity between 'stack' and 'unload': 0.18181818181818182
    # Similarity between 'stack' and 'deconstruct': 0.15384615384615385
    # Similarity between 'stack' and 'unstack': 0.0