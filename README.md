
#### Plan retrieval
This project has been developed with Python 3.12.
You can use [our fork](https://github.com/maayan25/LLMs-Planning) of PlanBench to get LLM-generated plans.

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

Run the evaluation script to run the full pipeline for your chosen model/task combinations:
```bash
cd scripts/plan_recovery
python3 summarise_results.py
./run_experiments.sh
```
Or run the evaluation script directly for a specific model/task combination:`
```bash
cd scripts/plan_recovery
python3 run_evaluation.py --model <model_name> --task <task_name>
```
