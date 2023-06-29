
from config import BaseConfig
from tqdm import tqdm
from sklearn.metrics import classification_report
from statistics import mode
from sentence_transformers import util
import torch
import random
from utils import build_test_samples, save_json
import argparse
from sentence_transformers import SentenceTransformer
import pandas as pd
 
random.seed(555)

def load_test(config, template):
    dataset_csv = pd.read_csv(config.subtask_train_test)
    source_text = build_test_samples(dataset_csv['text'].tolist(), template)
    target_text = dataset_csv['label'].tolist() 
    return source_text, target_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True) # 1, 2, 3
    parser.add_argument("--model", required=True)   # large, xl
    parser.add_argument("--sbert_fsl", required=True) 
    parser.add_argument("--use_sbert_fsl_combined", required=True)  
    parser.add_argument("--final_model", required=True)
    args = parser.parse_args()

    CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model=args.model)
    print("Evaluating Model:", CONFIG.model_input_path)
    print("Dataset    Path :", CONFIG.subtask_train_test)

    TEMPLATES = CONFIG.templates_dict['train']
    HYPOTHESIS_TEMPLATE = CONFIG.hypothesis_template
    
    if args.sbert_fsl=="True":
        model_output_path = CONFIG.sbert_model_output_path+"-fsl"
    elif args.use_sbert_fsl_combined=="True":
        model_output_path = CONFIG.sbert_model_output_path+"-fsl-combined"
    elif args.final_model == "True":
        model_output_path = CONFIG.sbert_model_output_path+"-final-fsl"
    else:
        model_output_path = CONFIG.sbert_model_output_path
        
    print("Testing Model Path:", model_output_path)
    MODEL = SentenceTransformer(model_output_path)
    
    CANDIDATE_LABELS = CONFIG.candidate_labels
    CANDIDATE_LABELS_EMBEDDING = MODEL.encode([HYPOTHESIS_TEMPLATE.format(label) for label in CANDIDATE_LABELS])
    prediction_dict = {}

    for index, template in enumerate(CONFIG.templates_dict['train']):
        TEMPLATES = template
        SOURCE, TARGET = load_test(CONFIG, TEMPLATES)
        predictions = []
        for source, target in tqdm(zip(SOURCE, TARGET)):
            source_embedding = MODEL.encode(source)
            predict_index = int(torch.argmax(util.cos_sim(source_embedding, CANDIDATE_LABELS_EMBEDDING)))
            predictions.append(CANDIDATE_LABELS[predict_index])
        prediction_dict[index] = predictions

    ensemble_predictions = []
    for index in range(len(TARGET)):
        ensemble_prediction = []
        for template_id, template_predicts in prediction_dict.items():
            for template_predict_index, template_predict in enumerate(template_predicts):
                if template_predict_index == index:
                    ensemble_prediction.append(template_predict)
                    break
        ensemble_predictions.append(ensemble_prediction)

    final_predictions = [mode(predictions) for predictions in ensemble_predictions]

    results = classification_report(TARGET, final_predictions, output_dict=True)
    print("F1-Score (Macro) is:", results['macro avg']['f1-score'])
    print("Storing results in:", CONFIG.result_multiple_file)
    report = {
        "results": results,
        "configs": vars(CONFIG)
    }
    save_json(report, CONFIG.result_multiple_file)

