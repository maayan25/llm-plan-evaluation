#!/bin/bash -l


echo "Starting Evaluation for HF models from $PWD"

STATUS_CSV="../../results/plan_recovery/final_evaluation_old/experiments_status.csv"
STALL_THRESHOLD=10 
STATE_FILE="logs/evaluation_old/combo_progress.csv" 
mkdir -p logs/evaluation_old

EXCLUDE=(
    # "logistics|gpt-oss-20b|_zero_shot"
    # "blocksworld|qwen3-32b|_state_tracking"
)

TASK_NAME_MAPPINGS=(
    "_zero_shot|_zero_shot"
    "_pddl|_pddl"
    "_state_tracking|_state_tracking"
    "_one_shot|"
)

is_excluded() {
    local combo="$1|$2|$3"
    for ex in "${EXCLUDE[@]}"; do
        [[ "$ex" == "$combo" ]] && return 0
    done
    return 1
}

map_task_name() {
    local csv_task="$1"
    for mapping in "${TASK_NAME_MAPPINGS[@]}"; do
        local csv_key="${mapping%%|*}"
        local arg_val="${mapping##*|}"
        if [[ "$csv_key" == "$csv_task" ]]; then
            echo "$arg_val"
            return 0
        fi
    done
    echo "ERROR: no task name mapping found for '$csv_task'" >&2
    return 1
}

get_state() {
    local key="$1"
    [[ -f "$STATE_FILE" ]] || return
    grep -F "${key}," "$STATE_FILE" | tail -n1
}

update_state() {
    local key="$1" last="$2" flag="$3"
    local tmp="${STATE_FILE}.tmp.$$"
    if [[ -f "$STATE_FILE" ]]; then
        grep -vF "${key}," "$STATE_FILE" > "$tmp"
    else
        : > "$tmp"
    fi
    echo "${key},${last},${flag}" >> "$tmp"
    mv "$tmp" "$STATE_FILE"
}

if [[ ! -f "$STATUS_CSV" ]]; then
    echo "ERROR: status CSV not found at '$STATUS_CSV'" >&2
    exit 1
fi

touch "$STATE_FILE"

combos=$(awk -F',' '
    NR==1 {
        for (i=1;i<=NF;i++) col[$i]=i
        if (!("domain" in col) || !("model" in col) || !("task" in col) || !("results_remaining" in col)) {
            print "MISSING_COLUMNS" > "/dev/stderr"
            exit 1
        }
        next
    }
    {
        rr  = $(col["results_remaining"])
        if (rr+0 > 10) {
            print $(col["domain"]) "," $(col["model"]) "," $(col["task"]) "," rr
        }
    }
' "$STATUS_CSV")

if [[ $? -ne 0 ]]; then
    echo "ERROR: '$STATUS_CSV' is missing one of the expected columns:" >&2
    echo "  domain, model, task, responses_remaining, results_remaining" >&2
    exit 1
fi

if [[ -z "$combos" ]]; then
    echo "No combinations found with results_remaining > 5. Nothing to do."
    exit 0
fi

# Run
while IFS=',' read -r domain model task results_remaining; do
    [[ -z "$domain" ]] && continue

    key="$domain|$model|$task"

    if is_excluded "$domain" "$model" "$task"; then
        echo "Skipping excluded combo: domain=$domain model=$model task=$task"
        continue
    fi

    state_line=$(get_state "$key")
    prev_remaining=""
    prev_flag="0"
    if [[ -n "$state_line" ]]; then
        prev_remaining=$(echo "$state_line" | cut -d',' -f2)
        prev_flag=$(echo "$state_line" | cut -d',' -f3)
    fi

    if [[ "$prev_flag" == "1" ]]; then
        echo "Skipping flagged (stalled) combo: domain=$domain model=$model task=$task -- reset it in $STATE_FILE to retry"
        continue
    fi

    if [[ -n "$prev_remaining" ]]; then
        progress=$(( prev_remaining - results_remaining ))
        if (( progress < STALL_THRESHOLD )); then
            echo "WARNING: domain=$domain model=$model task=$task made insufficient progress since last run (results_remaining was $prev_remaining, now $results_remaining, delta=$progress < $STALL_THRESHOLD). Flagging and skipping."
            update_state "$key" "$results_remaining" "1"
            continue
        fi
    fi

        task_arg=$(map_task_name "$task") || { echo "Skipping unmapped task: $task"; continue; }
        echo "Mapped task '$task' to argument '$task_arg'"

    start_time=$(date +%s)
    echo "Generating responses for domain: $domain, model: $model, task: $task (mapped to $task_arg)"
    timeout 7200 python3 run_evaluation.py --domain "$domain" --model "$model" --task "$task_arg" 2>&1 | tee "logs/$domain-$model-$task.log"
    exit_code=${PIPESTATUS[0]} # otherwise exit_code is from tee, not the python script
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [[ $exit_code -eq 124 ]]; then
        echo "TIMED OUT after ${duration}s: domain=${domain}, model=${model}, task=${task}"
    else
        echo "Duration for ${domain} ${task} ${model} is ${duration} seconds"
    fi

    update_state "$key" "$results_remaining" "0"
done <<< "$combos"

if [[ -f "$STATE_FILE" ]]; then
    flagged_lines=$(awk -F',' '$3 == "1" { print $1 }' "$STATE_FILE")
    if [[ -n "$flagged_lines" ]]; then
        echo ""
        echo "Flagged (stalled) combos:"
        echo "$flagged_lines" | sed 's/^/  /'
    fi
fi