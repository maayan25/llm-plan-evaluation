# script to run the python experiments consecutively


domains=("blocksworld_3" "blocksworld" "logistics") # "elevator") # "satellite")
#domains=("elevator" "satellite")
models=("claude-opus-4-8" "gemma4-12b" "gemma4-26b-a4b_it" "qwen3-8b") # "gemini_2-5_pro" "claude_3-5_haiku" "gemini_3-1_pro" "gpt-6-sol_chat"
#models=("gemini_3-8_flash"  "qwen3-14b") # "gemma4-26b-a4b_it" "qwen3-8b") #
tasks=("_pddl" "_zero_shot")
#
declare -A defaults=([top_k]=10 [min_k]=7 [max_k]=25 [refine_iters]=2 [tol]=0.1)
declare -A levels=(
    [top_k]="3 5 10 20 40"
    [max_k]="8 15 25 50"
    [refine_iters]="0 1 2 4"
    [tol]="0.05 0.1 0.2 0.4"
)


base=""
for p in "${!defaults[@]}"; do base+=" --$p ${defaults[$p]}"; done

configs=("$base")
for p in "${!levels[@]}"; do
    for v in ${levels[$p]}; do
        [ "$v" == "${defaults[$p]}" ] && continue     # already covered by baseline
        configs+=("$base --$p $v")
    done
done

mkdir -p logs

for cfg in "${configs[@]}"; do
    cfg_tag=$(echo "$cfg" | sed -E 's/--//g; s/[[:space:]]+/_/g' | sed -E 's/_([0-9.]+)/\1/g')

    for domain in "${domains[@]}"; do
        for model in "${models[@]}"; do
            for task in "${tasks[@]}"; do
                start_time=$(date +%s)
                echo "Running experiment for domain: $domain, model: $model, task: $task, cfg: $cfg"

                # $cfg is intentionally unquoted so it splits into separate arguments
                timeout 7200 python3 run_evaluation.py \
                    --domain "$domain" --model "$model" --task "$task" $cfg \
                    2>&1 | tee "logs/$domain-$model-$task-$cfg_tag.log"

                if [ "${PIPESTATUS[0]}" -eq 124 ]; then
                    echo "Timed out: $domain, $model, $task, $cfg_tag" >> logs/timeouts.log
                fi

                end_time=$(date +%s)
                echo "Duration for ${domain} ${task} ${model} [$cfg_tag] is $((end_time - start_time)) seconds"
            done
        done
    done
done