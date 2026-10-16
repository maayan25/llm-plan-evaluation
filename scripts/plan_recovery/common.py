# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# File for common functions used in the plan recovery scripts
import glob
import json
import os

from utils import parse_plan
from nltk.corpus import wordnet as wn
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
os.mkdir("cache") if not os.path.exists("cache") else None

_similarity_cache: dict[str, float] = {}
_synset_cache: dict[str, float] = {}
_meaningful_word_cache: dict[str, bool] = {}

SIMILARITY_CACHE_PATH = os.path.join(current_dir, "cache", "semantic_similarity_cache.json")
SYNSET_CACHE_PATH = os.path.join(current_dir, "cache", "synset_similarity_cache.json")
MEANINGFUL_WORD_CACHE_PATH = os.path.join(current_dir, "cache", "meaningful_word_cache.json")

def get_largest_subsequence(curr_plan, gt_plan) -> list[str]:
    """
    Get the largest subsequence of the ground truth plan that is in the current plan, from the gt plan that is closest
    to the current plan.
    A subsequence can be achieved by removing some or no elements from the lists without changing the order of the
    remaining elements.
    If the subsequence is of length of the GT plan, then potentially no regeneration is required.
    :param curr_plan: the generated plan to evaluate
    :param gt_plan: the ground truth plan to compare against
    :return: the size of the largest subsequence
    """
    if not curr_plan or not gt_plan:
        return []

    if curr_plan == gt_plan:
        return curr_plan

    # Space optimised dynamic programming, based on https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/
    m = len(curr_plan)
    n = len(gt_plan)
    matr = [[0] * (n + 1) for _ in range(m + 1)]
    subset = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if curr_plan[i - 1] == gt_plan[j - 1]:
                matr[i][j] = matr[i - 1][j - 1] + 1
                subset[i][j] = subset[i - 1][j - 1] + "\n" + curr_plan[i - 1]  # same structure for str as in PlanBench output...
            else:
                # matr[i][j] = max(matr[i - 1][j], matr[i][j - 1])
                if matr[i - 1][j] > matr[i][j - 1]:
                    matr[i][j] = matr[i - 1][j]
                    subset[i][j] = subset[i - 1][j]
                else:
                    matr[i][j] = matr[i][j - 1]
                    subset[i][j] = subset[i][j - 1]

    subseq = subset[m][n]
    # print(f"Subsequence computed: {subseq}, type: {type(subseq)}")
    return parse_plan(subseq)


def get_largest_subset(curr_plan, gt_plan) -> list[str]:
    """
    Get the largest subset of the ground truth plan that is in the current plan, from the gt plan that is closest
    to the current plan.
    A subset is only a contiguous sequence of elements from the list.
    :param curr_plan: the generated plan to evaluate
    :param gt_plan: the ground truth plan to compare against
    :return: the size of the largest subset
    """
    if not curr_plan or not gt_plan:
        return []

    if curr_plan == gt_plan:
        return curr_plan

    # Space optimised dynamic programming, based on https://www.geeksforgeeks.org/longest-common-substring-dp-29/
    m = len(curr_plan)
    n = len(gt_plan)
    matr = [[0] * (n + 1) for _ in range(m + 1)]

    longest_sub = 0
    end = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if curr_plan[i - 1] == gt_plan[j - 1]:
                matr[i][j] = matr[i - 1][j - 1] + 1
                if matr[i][j] > longest_sub:
                    longest_sub = matr[i][j]
                    end = i
            else:
                matr[i][j] = 0  # reset to 0

    start = end - longest_sub
    return curr_plan[start:end]


def parse_goal_state(problem):
    """
    Parse the goal state from the problem
    :param problem: the instance parsed with PDDLReader
    :return: the goal state as a list of atoms
    """
    # Check if goal is surrounded by parentheses
    goal = str(problem.goal)
    if goal.startswith("(") and goal.endswith(")"):
        goal = goal[1:-1].strip()
    parsed_goal = []

    predicates = goal.split(" and ")
    for pred in predicates:
        pred_name = pred.split("(")[0].strip()
        args_content = pred.split("(")[1].split(")")[0].strip()

        if args_content != "":
            args = " ".join(args_content.split(","))
            new_pred = f"({pred_name} {args})"
        else:
            new_pred = f"({pred_name})"

        parsed_goal.append(new_pred)
    # print(f"Parsed goal state: {parsed_goal}")
    return parsed_goal


def parse_init_state(problem):
    """
    Parse the initial state of the problem from the PDDLReader instance
    :param problem: the instance parsed with PDDLReader
    :return: the initial state as a list of strings
    """
    init = problem.init.as_atoms()
    parsed_init = []
    for pred in init:
        if isinstance(pred, tuple):
            # Numeric fluent assignment: (function_term, value)
            term, value = pred
            term_str = str(term)          # "travel-fast(n1, n2)"
            value_str = str(value).split(" ")[0]  # "4.0 (number)" -> "4.0"
            if value_str.endswith(".0"):
                value_str = value_str[:-2] # Remove ".0" for ints

            func_name = term_str.split("(")[0].strip()
            args_content = term_str.split("(")[1].split(")")[0].strip()

            if args_content != "":
                args = " ".join(a.strip() for a in args_content.split(","))
                new_pred = f"(= ({func_name} {args}) {value_str})"
            else:
                new_pred = f"(= ({func_name}) {value_str})"
        else:
            # Normal boolean atom
            pred = str(pred)
            pred_name = pred.split("(")[0]
            args_content = pred.split("(")[1].split(")")[0]

            if args_content != "":
                args = " ".join(args_content.split(","))
                new_pred = f"({pred_name} {args})"
            else:
                new_pred = f"({pred_name})"

        parsed_init.append(new_pred)
    return parsed_init

def get_variable_lists(actions: list[list[str]]) -> list[str]:
    """
    list of all the variables that are used in the given actions
    :param actions: a list of actions
    :return: a list of all the variables which are used in these actions
    """
    return list(dict.fromkeys(a for act in actions for a in act[1:]))

def make_list_hashable(obj):
    if isinstance(obj, list):
        return tuple(make_list_hashable(e) for e in obj)
    return obj


#### Semantic similarity functions
def best_synset_similarity(word1, word2) -> float:
    """
    :param word1: first word
    :param word2: second word
    :return: best Wu-Palmer similarity between synsets of two words.
    """
    key = "|".join(sorted((word1, word2)))
    if key in _synset_cache:
        return _synset_cache[key]

    synsets1 = wn.synsets(word1, pos=wn.VERB) + wn.synsets(word1, pos=wn.NOUN)
    synsets2 = wn.synsets(word2, pos=wn.VERB) + wn.synsets(word2, pos=wn.NOUN)
    best = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2)
            if sim is not None and sim > best:
                best = sim

    _synset_cache[key] = best
    return best

def normalise_action_name(action_name: str) -> str:
    """
    Normalises an action name and provides a syonym for actions not covered by WordNet.
    :param action_name: the action name to normalise
    :return: the normalised action name
    """
    SYNONYM_MAP = {
        "unstack": "remove",
    }
    norm = action_name.lower().replace("_", " ").replace("-", " ")
    return SYNONYM_MAP.get(norm, norm)

def calculate_semantic_similarity(act1, act2) -> float:
    """
    Computes semantic similarity between two action names.
    :param act1: first action name
    :param act2: second action name
    :return: similarity score between 0 and 1
    """
    key = _cache_key(act1, act2)
    if key in _similarity_cache:
        return _similarity_cache[key]

    a1 = act1.lower().replace("_", " ").replace("-", " ")
    a2 = act2.lower().replace("_", " ").replace("-", " ")

    # Try whole phrase first
    syn1 = wn.synsets(a1, pos=wn.VERB) + wn.synsets(a1, pos=wn.NOUN)
    syn2 = wn.synsets(a2, pos=wn.VERB) + wn.synsets(a2, pos=wn.NOUN)
    result = None
    if syn1 and syn2:
        sim = syn1[0].wup_similarity(syn2[0])
        if sim is not None and sim > 0:
            result = sim

    if result is None:
        words1, words2 = a1.split(), a2.split()
        scores = []
        for w1 in words1:
            best = 0.0
            for w2 in words2:
                sim = best_synset_similarity(w1, w2)
                if sim > best:
                    best = sim
            scores.append(best)
        result = float(np.mean(scores)) if scores else 0.0

    _similarity_cache[key] = result
    return result

def _cache_key(a: str, b: str) -> str:
    # order-independent key, since similarity(a, b) == similarity(b, a)
    return "|".join(sorted((a, b)))

def save_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

def load_all_caches():
    load_similarity_cache()
    load_synset_cache()
    load_meaningful_word_cache()

# Similarity cache
def load_similarity_cache():
    global _similarity_cache
    if os.path.exists(SIMILARITY_CACHE_PATH):
        with open(SIMILARITY_CACHE_PATH, "r") as f:
            _similarity_cache = json.load(f)
    else:
        _similarity_cache = {}

def save_similarity_cache():
    save_cache(_similarity_cache, SIMILARITY_CACHE_PATH)

def get_similarity_cache():
    return _similarity_cache

# Synset cache
def load_synset_cache():
    global _synset_cache
    if os.path.exists(SYNSET_CACHE_PATH):
        with open(SYNSET_CACHE_PATH, "r") as f:
            _synset_cache = json.load(f)
    else:
        _synset_cache = {}

def save_synset_cache():
    save_cache(_synset_cache, SYNSET_CACHE_PATH)

def get_synset_cache():
    return _synset_cache

# Meaningful word cache
def load_meaningful_word_cache():
    global _meaningful_word_cache
    if os.path.exists(MEANINGFUL_WORD_CACHE_PATH):
        with open(MEANINGFUL_WORD_CACHE_PATH, "r") as f:
            _meaningful_word_cache = json.load(f)
    else:
        _meaningful_word_cache = {}

def save_meaningful_word_cache():
    save_cache(_meaningful_word_cache, MEANINGFUL_WORD_CACHE_PATH)

def get_meaningful_word_cache():
    return _meaningful_word_cache

def is_meaningful_word(word: str) -> bool:
    if word in _meaningful_word_cache:
        return _meaningful_word_cache[word]
    result = bool(wn.synsets(word))
    _meaningful_word_cache[word] = result
    return result

# For parallel processing
def save_worker_caches():
    os.makedirs("cache/partial", exist_ok=True)
    pid = os.getpid()
    save_cache(get_similarity_cache(), f"cache/partial/similarity_{pid}.json")
    save_cache(get_synset_cache(), f"cache/partial/synset_{pid}.json")
    save_cache(get_meaningful_word_cache(), f"cache/partial/meaningful_{pid}.json")

def merge_worker_caches():
    for prefix, path, loader in [
        ("similarity", SIMILARITY_CACHE_PATH, load_similarity_cache),
        ("synset", SYNSET_CACHE_PATH, load_synset_cache),
        ("meaningful", MEANINGFUL_WORD_CACHE_PATH, load_meaningful_word_cache),
    ]:
        merged = {}
        if os.path.exists(path):
            with open(path) as f:
                merged.update(json.load(f))
        for fpath in glob.glob(f"cache/partial/{prefix}_*.json"):
            with open(fpath) as fh:
                merged.update(json.load(fh))
        with open(path, "w") as f:
            json.dump(merged, f)
        for fpath in glob.glob(f"cache/partial/{prefix}_*.json"):
            os.remove(fpath)
