#!/bin/bash
python3 inference_fsl.py --subtask=1 --input=dataset/test_text/subtask1/test_text.json --output=output/run2_subtask1.json
python3 inference_fsl.py --subtask=2 --input=dataset/test_text/subtask2/test_text.json --output=output/run2_subtask2.json
python3 inference_fsl.py --subtask=3 --input=dataset/test_text/subtask3/test_text.json --output=output/run2_subtask3.json
