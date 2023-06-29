#!/bin/bash
python3 inference_fsl_biencoder.py --subtask=1 --input=dataset/test_text/subtask1/test_text.json --output=output/run1_subtask1.json
python3 inference_fsl_biencoder.py --subtask=2 --input=dataset/test_text/subtask2/test_text.json --output=output/run1_subtask2.json
python3 inference_fsl_biencoder.py --subtask=3 --input=dataset/test_text/subtask3/test_text.json --output=output/run1_subtask3.json
