#!/bin/bash
tasks=("1" "2" "3")

loglarge="random-baseline.log.txt"
exec > $loglarge

for task in "${tasks[@]}"; do
    echo "Run Test for Model: RandomModel on SubTask=$task:"
    python3 random_baseline.py --subtask=$task
    echo "-----------------------------------------------------------"
done
