# script to run the python experiments consecutively

models=("qwen3-8b" "qwen3-14b")
domains=("blocksworld" "elevator")
tasks=("_zero_shot")

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for domain in "${domains[@]}"; do
            start_time=$(date +%s)
            echo "Running experiment for domain: $domain, model: $model, task: $task"
            timeout 7200 python3 run_evaluation.py --domain "$domain" --model "$model" --task "$task" 2>&1 | tee "logs/$domain-$model-$task.log"
            if [ $? -eq 124 ]; then
                echo "Timed out: $domain, $model, $task" >> logs/timeouts.log
            fi
            end_time=$(date +%s)
            echo "Duration for ${domain} ${task} ${model} is $((end_time - start_time)) seconds"
        done
    done
done