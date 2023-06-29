#!/bin/bash
tasks=("1" "2" "3")
model="large"
combined_model="False"
final_model="True"
loglarge="train_test_runner-large.log.txt"
exec > >(tee -a "$loglarge")


for task in "${tasks[@]}"; do
    echo "----------------------------------------------------------------------------------------------------------------------"
    echo "SubTask $task"
    echo "----------------------------------------------------------------------------------------------------------------------"
    echo "Run Train for Model: --subtask=$task --model=$model --combined_model=$combined_model --final_model=$final_model"
    python3 train_flan_t5.py --subtask=$task --model=$model --combined_model=$combined_model --final_model=$final_model
    echo " "
    echo " "
    echo "Run Test for Model: --subtask=$task --model=$model --combined_model=$combined_model --final_model=$final_model"
    python3 test_flan_t5.py --subtask=$task --model=$model --combined_model=$combined_model --final_model=$final_model
    echo "----------------------------------------------------------------------------------------------------------------------"
done