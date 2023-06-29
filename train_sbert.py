
from config import BaseConfig
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from torch.utils.data import DataLoader
from utils import cleaner
import random
import argparse

random.seed(555)

def generate_negative_samples(dataset_csv, hypothesis_template, negative_samples_no):
    dataset_dict = {}
    for text, label in zip(dataset_csv['text'].tolist(), dataset_csv['label'].tolist()):
        if label in dataset_dict:
            dataset_dict[label].append(text)
        else:
            dataset_dict[label] = [text]
            
    dataset_ng = {'text':[], 'ap-label':[], 'ng-label':[]}
    for positive_label, positive_texts in dataset_dict.items():
        for text in positive_texts:
            dataset_ng['text'].append(text)
            dataset_ng['ap-label'].append(hypothesis_template.format(positive_label))
            dataset_ng['ng-label'].append(1)
        for negative_label, negative_texts in dataset_dict.items():
            if negative_label != positive_label:
                negative_candidates_texts = random.sample(negative_texts, negative_samples_no)
                for text in negative_candidates_texts:
                    dataset_ng['text'].append(text)
                    dataset_ng['ap-label'].append(hypothesis_template.format(negative_label))
                    dataset_ng['ng-label'].append(0)   
    return dataset_ng

def build_sbert_train_samples(texts, ap_labels, ng_labels, templates):
    cl_dataset, mnr_dataset, softmax_dataset = [], [], []
    for text, ap_label, ng_label in zip(texts, ap_labels, ng_labels):
        cleaned_tweet = cleaner(text)
        for template in templates:
            cl_dataset.append(InputExample(texts=[template.replace("{tweets}", cleaned_tweet), ap_label], 
                                           label=ng_label))
            softmax_dataset.append(InputExample(texts=[template.replace("{tweets}", cleaned_tweet), ap_label], 
                                                label=ng_label))
            if ng_label == 1:
                mnr_dataset.append(InputExample(texts=[template.replace("{tweets}", cleaned_tweet), ap_label], 
                                                label=ng_label))
                mnr_dataset.append(InputExample(texts=[ap_label, template.replace("{tweets}", cleaned_tweet)], 
                                                label=ng_label))
    return cl_dataset, mnr_dataset, softmax_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True) # 1, 2, 3
    parser.add_argument("--model", required=True)   # large, xl
    parser.add_argument("--sbert_fsl", required=True)
    parser.add_argument("--use_sbert_fsl_combined", required=True)  
    parser.add_argument("--final_model", required=True)

    args = parser.parse_args()
    
    CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model=args.model)
    # print("Evaluating Model:", CONFIG.model_input_path)
    print("Dataset    Path :", CONFIG.subtask_train_train)
    print("Vars(args):", vars(args))
    TEMPLATES = CONFIG.templates_dict['train']
    HYPOTHESIS_TEMPLATE = CONFIG.hypothesis_template

    NEGATIVE_SAMPLES_NO = CONFIG.tasks_stats["NG"]

    CANDIDATE_LABELS = CONFIG.candidate_labels
    if args.final_model == 'True':
        CONFIG.subtask_train_train = CONFIG.subtask_train

    DATASET = pd.read_csv(CONFIG.subtask_train_train)

    DATASET = generate_negative_samples(DATASET, HYPOTHESIS_TEMPLATE, NEGATIVE_SAMPLES_NO)
    CL_DATASET, MNRL_DATASET, _ = build_sbert_train_samples(texts=DATASET['text'],
                                                            ap_labels=DATASET['ap-label'],
                                                            ng_labels=DATASET['ng-label'],
                                                            templates=TEMPLATES)

    print(f"size of CL:{len(CL_DATASET)} Size of MNRL:{len(MNRL_DATASET)}")
    if args.sbert_fsl == "True":
        model_input_path =  CONFIG.model_output_path
        model_output_path = CONFIG.sbert_model_output_path+"-fsl"
    elif args.use_sbert_fsl_combined == "True":
        model_input_path =  CONFIG.model_output_path_combined
        model_output_path = CONFIG.sbert_model_output_path+"-fsl-combined"
    elif args.final_model == "True":
        print("Training model for: FINAL FSL TRAINING FOR SUBMITING")
        model_input_path =  CONFIG.model_output_path+"-final"
        model_output_path = CONFIG.sbert_model_output_path+"-final-fsl"
    else:
        model_input_path = CONFIG.model_input_path
        model_output_path = CONFIG.sbert_model_output_path
    
    print("Training Model Path:", model_input_path)
    MODEL = SentenceTransformer(model_input_path)

    DATALOADER_MNRL = DataLoader(MNRL_DATASET, shuffle=True, batch_size=CONFIG.sbert_batch_size)
    MNRL = losses.MultipleNegativesRankingLoss(MODEL)

    DATALOADER_CL = DataLoader(CL_DATASET, shuffle=True, batch_size=CONFIG.sbert_batch_size)
    CL = losses.OnlineContrastiveLoss(model=MODEL,
                                     distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, 
                                     margin=CONFIG.sbert_margin)

    OBJECTIVES = [(DATALOADER_CL, CL), (DATALOADER_MNRL, MNRL)]

    MODEL.fit(train_objectives=OBJECTIVES,
             epochs=CONFIG.sbert_num_epochs,
             warmup_steps=1000,
             output_path=model_output_path)

    print("SAVING MODEL ..... ")
    MODEL.save(model_output_path)
    print("MODEL trained and saved into:", model_output_path)
