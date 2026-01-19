# script to run the python experiments consecutively

domains=("obfuscated_randomized_logistics") # blocksworld_3 logistics
#models=("llama" "qcode" "claude_3_haiku" "gemini_2_flash" "gpt-3.5-turbo_chat" "dsqwen" "claude_3-7_sonnet" "gemini_2-5_flash" "o3-mini_chat" "o4-mini_chat") # "qwen" "qwenl"
models=("o4-mini_chat") # After fix: "o3-mini_chat" "o4-mini_chat" zero-shot pddl
#tasks=("_zero_shot") # Finish "claude_3-7_sonnet"
tasks=("" "_pddl" "_zero_shot")

mkdir -p logs

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for domain in "${domains[@]}"; do
            start_time=$(date +%s)
            echo "Running experiment for domain: $domain, model: $model, task: $task"
            timeout 3000 python run_evaluation.py --domain "$domain" --model "$model" --task "$task" &> "logs/$domain-$model-$task.log"
            if [ $? -eq 124 ]; then
                echo "Timed out: $domain, $model, $task" >> logs/timeouts.log
            fi
            end_time=$(date +%s)
            echo "Duration for ${domain} ${task} ${model} is $((end_time - start_time)) seconds"
        done
    done
done

#domains=("logistics") # blocksworld_3 logistics
#models=("qwen" "qwenl" "llama" "qcode" "qcodel" "claude_3_haiku" "gemini_2_flash" "gpt-3.5-turbo_chat" "dsqwen" "claude_3-7_sonnet" "gemini_2-5_flash") # "o3-mini_chat" "o4-mini_chat"
##models=("gemini_2-5_flash") # After fix: "o3-mini_chat" "o4-mini_chat" 50-100
#tasks=("_state_tracking") # Finish "claude_3-7_sonnet"
##tasks=("" "_state_tracking" "_pddl" "_zero_shot")
#
#for task in "${tasks[@]}"; do
#    for model in "${models[@]}"; do
#        for domain in "${domains[@]}"; do
#            start_time=$(date +%s)
#            echo "Running experiment for domain: $domain, model: $model, task: $task"
#            timeout 3000 python run_evaluation.py --domain "$domain" --model "$model" --task "$task" &> "logs/$domain-$model-$task.log"
#            if [ $? -eq 124 ]; then
#                echo "Timed out: $domain, $model, $task" >> logs/timeouts.log
#            fi
#            end_time=$(date +%s)
#            echo "Duration for ${domain} ${task} ${model} is $((end_time - start_time)) seconds"
#        done
#    done
#done