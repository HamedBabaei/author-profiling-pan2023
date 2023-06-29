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
# from torch.nn.functional import softmax

def process_nli(premise: str, hypothesis: str):
    """ process to required xnli format with task prefix """
    return "".join(['xnli: premise: ', premise, ' hypothesis: ', hypothesis])

def load_test(config, template):
    dataset_csv = pd.read_csv(config.subtask_train_test)
    source_text = build_test_samples(dataset_csv['text'].tolist(), template)
    target_text = dataset_csv['label'].tolist()
    return source_text, target_text
 
if __name__ == "__main__":
    CANDIDATE_LABELS = {1:list(label_mapper.SUBTASK_1.keys()), 
                        2:list(label_mapper.SUBTASK_2.keys()),
                        3:list(label_mapper.SUBTASK_3.keys())}
    HYPOTHESIS_TEMPLATE = {1:"This user profile in cryptocurrency is a {}",
                           2:"This influencer interest is a {}",
                           3:"This influencer intent is a {}"}
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True) # 1, 2, 3
    parser.add_argument("--model", required=True)   # large, xl
    args = parser.parse_args()

    CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model=args.model)
    print("Evaluating Model:", CONFIG.model_input_path)
    print("Dataset    Path :", CONFIG.subtask_train_test)
    
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.model_input_path)
    MODEL = T5ForConditionalGeneration.from_pretrained(CONFIG.model_input_path)
    MODEL.to(CONFIG.device)

    MODEL.eval()
    LABEL_INDEXES = TOKENIZER.convert_tokens_to_ids([f"_{str(index)}" 
                    for index,_ in enumerate(CANDIDATE_LABELS[int(args.subtask)])])
    prediction_dict = {}

    for index, template in enumerate(CONFIG.templates_dict['train']):
        TEMPLATES = template
        SOURCE, TARGET = load_test(CONFIG, TEMPLATES)
        predictions = []
        for source, target in tqdm(zip(SOURCE, TARGET)):
            pairs = [(source, HYPOTHESIS_TEMPLATE[int(args.subtask)].format(label))
                    for label in CANDIDATE_LABELS[int(args.subtask)]]
            seqs = [process_nli(premise=premise, hypothesis=hypothesis) 
                    for premise, hypothesis in pairs]
            inputs = TOKENIZER.batch_encode_plus(seqs, 
                                                max_length=CONFIG.max_source_length, 
                                                padding='max_length', 
                                                return_tensors="pt", 
                                                truncation=True)
            inputs.to(CONFIG.device)
            
            out = MODEL.generate(**inputs, 
                                output_scores=True, 
                                return_dict_in_generate=True,
                                num_beams=1)
            scores = out.scores[0]
            scores = scores[:, LABEL_INDEXES[0]]
            predict = CANDIDATE_LABELS[int(args.subtask)][int(torch.argmax(scores).cpu())]
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
