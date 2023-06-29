from utils import build_test_samples, save_json
from config import BaseConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import argparse
import pandas as pd
from statistics import mode
import label_mapper
import random

random.seed(0)

def load_test(config, template):
    dataset_csv = pd.read_csv(config.subtask_train_test)
    source_text = build_test_samples(dataset_csv['text'].tolist(), template)
    target_text = dataset_csv['label'].tolist()
    return source_text, target_text

if __name__ == "__main__":
    CANDIDATE_LABELS = {1:list(label_mapper.SUBTASK_1.keys()), 
                        2:list(label_mapper.SUBTASK_2.keys()),
                        3:list(label_mapper.SUBTASK_3.keys())}

    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True) # 1, 2, 3
    args = parser.parse_args()

    CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model="NONE")
    print("Evaluating Model:", CONFIG.model_input_path)
    print("Dataset    Path :", CONFIG.subtask_train_test)

    TEMPLATES = CONFIG.templates_dict['test']
    SOURCE, TARGET = load_test(CONFIG, TEMPLATES)
    predictions = []
    for source, target in tqdm(zip(SOURCE, TARGET)):
        predict = random.choice(CANDIDATE_LABELS[int(args.subtask)])
        predictions.append(predict)
        
    results = classification_report(TARGET, predictions, output_dict=True)
    print("F1-Score (Macro) is:", results['macro avg']['f1-score'])
    print("Storing results in:", CONFIG.result_multiple_file)
    report = {
        "results": results,
        "configs": vars(CONFIG)
    }
    save_json(report, CONFIG.result_multiple_file)
