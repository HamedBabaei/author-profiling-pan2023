#!/bin/bash
tasks=("1" "2" "3")
models=("large")

loglarge="zero-shot-baseline-flan-t5-large.log.txt"
exec > $loglarge

for task in "${tasks[@]}"; do
    echo "SubTask $task"
    for model in  "${models[@]}"; do
        echo "Run Test for Model: Flan-T5-$model on SubTask=$task:"
        python3 zero_shot_baseline.py --subtask=$task --model=$model
        echo "-----------------------------------------------------------"
    done
done
