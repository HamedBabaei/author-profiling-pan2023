from utils import build_test_samples, save_json
from config import BaseConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
import argparse
import pandas as pd
from statistics import mode

def predict(X, model, tokenizer, max_source_length, max_target_length, device):
    inputs = tokenizer(X, max_length=max_source_length, padding='max_length', 
                       return_tensors="pt", truncation=True)
    inputs.to(device)
    with torch.no_grad():
        sequence_ids = model.generate(inputs.input_ids,num_beams=1, max_length=max_target_length)
    sequences = tokenizer.batch_decode(sequence_ids, skip_special_tokens=True)
    return sequences

def load_test(config, template):
    dataset_csv = pd.read_csv(config.subtask_train_test)
    source_text = build_test_samples(dataset_csv['text'].tolist(), template)
    target_text = dataset_csv['label'].tolist()
    return source_text, target_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True) # 1, 2, 3
    parser.add_argument("--model", required=True)   # large, xl
    parser.add_argument("--combined_model", required=True)  
    parser.add_argument("--final_model", required=True) 
    args = parser.parse_args()

    CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model=args.model)
    print("ARGS:", vars(args))
    print("Evaluating Model:", CONFIG.model_output_path)
    print("Dataset    Path :", CONFIG.subtask_train_test)
    if args.combined_model == "True":
        print("Testing model for: COMBINED MODELS (MULTI-TASK FSL LEARNING)")
        TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.model_output_path_combined)
        MODEL = T5ForConditionalGeneration.from_pretrained(CONFIG.model_output_path_combined)
    elif args.final_model=="True":
        print("Testing model for: FINAL FSL TRAINING FOR SUBMITING")
        TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.model_output_path+"-final")
        MODEL = T5ForConditionalGeneration.from_pretrained(CONFIG.model_output_path+"-final")
    else:
        print("Testing model for: SINGLE TASK FSL LEARNING")
        TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.model_output_path)
        MODEL = T5ForConditionalGeneration.from_pretrained(CONFIG.model_output_path)
    MODEL.to(CONFIG.device)
    MODEL.eval()

    prediction_dict = {}

    for index, template in enumerate(CONFIG.templates_dict['train']):
        TEMPLATES = template
        SOURCE, TARGET = load_test(CONFIG, TEMPLATES)
        predictions = []
        prediction_error_dict =  {'no influence':'no influencer', 
                                'racing':'gaming', 
                                'sleeping':'other', 
                                'lastgame':'gaming'}
        for source, target in tqdm(zip(SOURCE, TARGET)):
            inputs = TOKENIZER(source, 
                            max_length=CONFIG.max_source_length, 
                            padding='max_length', 
                            return_tensors="pt", 
                            truncation=True)
            inputs.to(CONFIG.device)
            with torch.no_grad():
                sequence_ids = MODEL.generate(inputs.input_ids,
                                            num_beams=1, 
                                            max_length=CONFIG.max_target_length)
            predict = TOKENIZER.batch_decode(sequence_ids, skip_special_tokens=True)[0]

            predict = prediction_error_dict.get(predict, predict)
            if predict in target:
                predict = target
            predictions.append(predict)
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
