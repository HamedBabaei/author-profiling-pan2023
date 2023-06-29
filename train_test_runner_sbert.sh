#!/bin/bash
tasks=("1" "2" "3")
model="large"
sbert_fsl="False"
use_sbert_fsl_combined="False"
final_model="True"
loglarge="train_test_runner_sbert.log.txt"
exec > >(tee -a "$loglarge")

for task in "${tasks[@]}"; do
    echo "----------------------------------------------------------------------------------------------------------------------"
    echo "SubTask $task"
    echo "----------------------------------------------------------------------------------------------------------------------"
    echo "Run Train for Model: --subtask=$task --model=$model --sbert_fsl=$sbert_fsl --use_sbert_fsl_combined=$use_sbert_fsl_combined --final_model=$final_model"
    python3 train_sbert.py --subtask=$task --model=$model --sbert_fsl=$sbert_fsl --use_sbert_fsl_combined=$use_sbert_fsl_combined --final_model=$final_model
    echo " "
    echo " "
    echo "Run Test for Model: --subtask=$task --model=$model --sbert_fsl=$sbert_fsl --use_sbert_fsl_combined=$use_sbert_fsl_combined --final_model=$final_model"
    python3 test_sbert.py --subtask=$task --model=$model --sbert_fsl=$sbert_fsl --use_sbert_fsl_combined=$use_sbert_fsl_combined --final_model=$final_model
    echo "----------------------------------------------------------------------------------------------------------------------"
done
