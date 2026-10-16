# script to run the python experiments consecutively for the baseline

domains=("blocksworld_3" "blocksworld" "logistics" "elevator" "satellite")
tasks=("_pddl") # does not change GT
model="baseline"

mkdir -p logs

for task in "${tasks[@]}"; do
        for domain in "${domains[@]}"; do
            start_time=$(date +%s)
            echo "Running experiment for domain: $domain, task: $task"
            timeout 7200 python3 run_evaluation.py --domain "$domain" --model "$model" --task "$task" --baseline 2>&1 | tee "logs/baseline_$domain-$task.log"
            if [ $? -eq 124 ]; then
                echo "Timed out: $domain, $model, $task" >> logs/timeouts.log
            fi
            end_time=$(date +%s)
            echo "Duration for ${domain} ${task} ${model} is $((end_time - start_time)) seconds"
        done
done