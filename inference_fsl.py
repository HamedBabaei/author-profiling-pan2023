import argparse
import re
import templates
import label_mapper
from sentence_transformers import SentenceTransformer,  util
import torch
from statistics import mode
import json
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np

TEMPLATES_DICT = {
    "1": templates.SUBTASK_1,
    "2": templates.SUBTASK_2,
    "3": templates.SUBTASK_3
}

LABELD2ID_MAPPER = {
    "1": label_mapper.SUBTASK_1_LABEL2ID,
    "2": label_mapper.SUBTASK_2_LABEL2ID,
    "3": label_mapper.SUBTASK_3_LABEL2ID
}

CANDIDATE_LABELS = {
    "1":list(label_mapper.SUBTASK_1.keys()), 
    "2":list(label_mapper.SUBTASK_2.keys()),
    "3":list(label_mapper.SUBTASK_3.keys())
}

HYPOTHESIS_TEMPLATES = templates.HYPOTHESIS_TEMPLATE

SEP_TOKEN = "<SEP>"

def cleaner(text):
    text = "  ".join(text.split(SEP_TOKEN))
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+", "", text)
    text = text.replace("$", "")
    text = text.replace("@", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace(":", "")
    text = text.strip()
    text = text.lower()
    return text

def load_file(text_path):
    user_id=[]
    user_text=[]
    with open(text_path, 'r') as inp:
        for i in inp:
            tmp=json.loads(i)
            user_id.append(tmp['twitter user id'])
            texts = f"{SEP_TOKEN}".join([text['text'] for text in tmp['texts']])
            user_text.append(cleaner(texts))
    return user_id, user_text

def save_outputs(output_file, user_id, user_label, user_probs):
    df_output = pd.DataFrame(list(zip(user_id, user_label, user_probs)), columns =['twitter user id', 'class', 'probability'])
    df_output.to_json(output_file, orient='records', lines=True)
    df_output.to_csv(output_file+".csv", index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", type=int, help='The name of task to be performed on', required=True) 
    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)
    args = parser.parse_args()
    
    # set hypterparameters
    subtask_id = args.subtask
    candidate_labels = CANDIDATE_LABELS[str(subtask_id)]
    hypothesis_template = HYPOTHESIS_TEMPLATES[subtask_id]
    input_path = args.input
    output_path = args.output
    max_source_length = 512
    max_target_length = 3
    device = 'cpu'

    # create model input path based on given subtask and  load model
    model_input_path = f"assets/subtask{str(subtask_id)}-large-final"
    print(f"Model input path is:{model_input_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_input_path)
    model = T5ForConditionalGeneration.from_pretrained(model_input_path)
    model.to(device)
    model.eval()
    # load input data
    user_id, user_texts = load_file(input_path)    

    prediction_error_dict =  {'no influence':'no influencer',  'racing':'gaming', "shipping": "trading matters", "noo": 'no influencer',
                              'sleeping':'other', "voting":"other", 'lastgame':'gaming', "business": "trading matters"}

    user_predictions, user_predictions_probs = [], []
    for user_tweets in tqdm(user_texts):
        predicts, probas = [], {}
        for template in TEMPLATES_DICT[str(subtask_id)]['train']:
            user_tweets_new = template.replace("{tweets}", user_tweets)
            
            inputs = tokenizer(user_tweets_new, 
                            max_length=max_source_length, 
                            padding='max_length', 
                            return_tensors="pt", 
                            truncation=True)
            inputs.to(device)
            with torch.no_grad():
                sequence_ids = model.generate(inputs.input_ids,
                                              num_beams=1, 
                                              max_length=max_target_length)
            

            predict_gen = tokenizer.batch_decode(sequence_ids, skip_special_tokens=True)[0].lower()
            predict_gen = prediction_error_dict.get(predict_gen, predict_gen)
            predict = predict_gen
            for candidate in candidate_labels:
                if (candidate in predict_gen) or (predict_gen in candidate):
                    predict = candidate
            if predict == " ":
                pring("UNKNOWN label:", predict_gen)
                predict = predict_gen
            if predict in probas:
                probas[predict] += 1
            else:
                probas[predict] = 1

        for candidate, proba in probas.items():
            probas[candidate] = proba/10
        ensemble_predict_class = max(probas, key=probas.get)
        ensemble_predict_proba = probas[ensemble_predict_class]

        user_predictions.append(ensemble_predict_class)
        user_predictions_probs.append(ensemble_predict_proba)
    
    save_outputs(output_file=output_path, 
                 user_id=user_id, 
                 user_label=user_predictions, 
                 user_probs=user_predictions_probs)
