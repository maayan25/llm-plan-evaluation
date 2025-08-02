#!/bin/bash

# This script runs the analysis for the evaluation of LLMs on planning tasks.

domains=("" "blocksworld" "logistics")
success_rate=("" "high_success" "low_success")
missing_instances=(True)

for is_missing in "${missing_instances[@]}"; do
    for rate in "${success_rate[@]}"; do
        for domain in "${domains[@]}"; do
            start_time=$(date +%s)
            echo "Running analysis for domain: $domain, model: $rate, task: $is_missing"
            python analyse_results.py --domain "$domain" --success_rate "$rate" &> "logs/$domain-missing_instances-$rate.log"
            end_time=$(date +%s)
            echo "Duration for ${domain} ${rate} ${is_missing} is $((end_time - start_time)) seconds"
        done
    done
done
