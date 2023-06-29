import argparse
import datetime
import os
import torch
import templates
import label_mapper

class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.templates_dict = {
            "1": templates.SUBTASK_1,
            "2": templates.SUBTASK_2,
            "3": templates.SUBTASK_3
        }
        self.label2id_mapper = {
            "1": label_mapper.SUBTASK_1_LABEL2ID,
            "2": label_mapper.SUBTASK_2_LABEL2ID,
            "3": label_mapper.SUBTASK_3_LABEL2ID
        }
        self.tasks_stats = {
            "1": {"N": 16, "C": 5, "NG": 15},
            "2": {"N": 32, "C": 5, "NG": 20},
            "3": {"N": 32, "C": 4, "NG": 20}
        }
        self.candidate_labels = {"1":list(label_mapper.SUBTASK_1.keys()), 
                                 "2":list(label_mapper.SUBTASK_2.keys()),
                                 "3":list(label_mapper.SUBTASK_3.keys())}

    def get_args(self, subtask: int, model:str):
        subtask=str(subtask)
        self.parser.add_argument("--subtask", type=int)
        self.parser.add_argument("--model", type=str)
        self.parser.add_argument("--templates_dict", type=dict, default=self.templates_dict[subtask])
        self.parser.add_argument("--hypothesis_template", type=str, default=templates.HYPOTHESIS_TEMPLATE[int(subtask)])
        self.parser.add_argument("--tasks_stats", type=dict, default=self.tasks_stats[subtask])
        self.parser.add_argument("--candidate_labels", type=dict, default=self.candidate_labels[subtask])
        self.parser.add_argument("--label2id_mapper", type=dict, default=self.label2id_mapper[subtask])

        self.parser.add_argument("--subtask_text", type=str, default=f"dataset/subtask{subtask}/train_text.json")
        self.parser.add_argument("--subtask_truth", type=str, default=f"dataset/subtask{subtask}/train_truth.json")
        
        self.parser.add_argument("--subtask_train", type=str, default=f"dataset/subtask{subtask}/subtask{subtask}_train_all.csv")
        self.parser.add_argument("--subtask_train_train", type=str, default=f"dataset/subtask{subtask}/subtask{subtask}_train_train.csv")
        self.parser.add_argument("--subtask_train_test", type=str, default=f"dataset/subtask{subtask}/subtask{subtask}_train_test.csv")
        
        self.parser.add_argument("--seed", type=int, default=555)
        self.parser.add_argument("--test_size_subtask1", type=int, default=16, help="number of samples per class for test split")
        self.parser.add_argument("--test_size_subtask23", type=int, default=32, help="number of samples per class for test split")
        self.parser.add_argument("--max_source_length", type=int, default=512)
        self.parser.add_argument("--max_target_length", type=int, default=3)
        self.parser.add_argument("--auto_find_batch_size", type=bool, default=True)
        self.parser.add_argument("--learning_rate", type=float, default=1e-5)
        self.parser.add_argument("--num_train_epochs", type=int, default=10)

        self.parser.add_argument("--label_pad_token_id", type=int, default=-100)
        self.parser.add_argument("--device", type=str, default='cuda')

        time = str(datetime.datetime.now()).split('.')[0]
        self.parser.add_argument("--model_input_path", type=str, default=f"../LLMs4OL/assets/LLMs/flan-t5-{model}")
        self.parser.add_argument("--model_output_path", type=str,  default=f"assets/subtask{subtask}-{model}")
        self.parser.add_argument("--model_output_path_combined", type=str,  default=f"assets/subtask-{model}-combined")
        self.parser.add_argument("--combined_model", type=str)
        self.parser.add_argument("--final_model", type=str)

        self.parser.add_argument("--output_log_dir", type=str,  default=f"assets/subtask{subtask}-{model}-log")            
        self.parser.add_argument("--result_file", type=str, default=f"results/Flan-T5-{model.upper()}-subtask-{subtask}-{time}.json")
        self.parser.add_argument("--result_multiple_file", type=str, default=f"results/Flan-T5-{model.upper()}-multple-subtask-{subtask}-{time}.json")

        self.parser.add_argument("--sbert_fsl", type=str)
        self.parser.add_argument("--use_sbert_fsl_combined", type=str)
        self.parser.add_argument("--sbert_model_output_path", type=str,  default=f"assets/sbert-subtask{subtask}-{model}")
        self.parser.add_argument("--sbert_num_epochs", type=int, default=10)
        self.parser.add_argument("--sbert_margin", type=int, default=0.5)
        self.parser.add_argument("--sbert_batch_size", type=int, default=2)

        self.parser.add_argument("-f")
        return self.parser.parse_args()

