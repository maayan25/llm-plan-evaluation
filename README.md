
#### Plan retrieval
This project has been developed with Python 3.12

#### Setup instructions
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required packages:
```bash
python3 -m pip install -r requirements.txt
```

#### Run the evaluation script
Download Wordnet for semantic similarity:
```bash
python3
> import nltk
> nltk.download('wordnet')
```
Modify path and configuration:
1. path to LLM-Planning-PlanBench repository in `scripts/plan-recovery/check_plan_validity.py`
2. name of model used for generation
3. name of results file

Run the evaluation script to run the full pipeline on all model/task combinations:
```bash
cd scripts/plan_recovery
./run_experiments.sh
```
Or run the evaluation script directly for a specific model/task combination:`
```bash
cd scripts/plan_recovery
python3 run_evaluation.py --model <model_name> --task <task_name>
```