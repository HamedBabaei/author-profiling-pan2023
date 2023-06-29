from config import BaseConfig
from utils import build_train_samples
import pandas as pd
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,\
                         Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, TrainerState
import argparse


def preprocess_function(sample, padding="max_length"):
    inputs = [item for item in sample["text"]]
    model_inputs = TOKENIZER(inputs, max_length=CONFIG.max_source_length, padding=padding, truncation=True)
    labels = TOKENIZER(text_target=sample["label"], max_length=CONFIG.max_target_length, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != TOKENIZER.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_train(config, templates):
    dataset_csv = pd.read_csv(config.subtask_train_train)
    source_text, target_text = build_train_samples(dataset_csv['text'].tolist(),
                                                   dataset_csv['label'].tolist(),
                                                   templates)
    dataset = DatasetDict({'train': Dataset.from_dict({'label': target_text, 'text': source_text})})
    return dataset

def load_combined_model_dataset(model, final_model):
    def load_train_local(CONFIG, templates):
        dataset_csv = pd.read_csv(config.subtask_train_train)
        source_text, target_text = build_train_samples(dataset_csv['text'].tolist(),
                                                    dataset_csv['label'].tolist(),
                                                    templates)
        return source_text, target_text

    source, target = [], []
    for task in [1, 2, 3]:
        CONFIG = BaseConfig().get_args(subtask=task, model=model)
        if final_model == "True":
            CONFIG.subtask_train_train = CONFIG.subtask_train
        templates = CONFIG.templates_dict['train']
        source_text, target_text = load_train_local(CONFIG, templates)
        source += source_text
        target += target_text
    TRAIN_DATASET = DatasetDict({'train': Dataset.from_dict({'label': target, 'text': source})})
    return TRAIN_DATASET, CONFIG

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True) # 1, 2, 3
    parser.add_argument("--model", required=True)   # large, xl
    parser.add_argument("--combined_model", required=True) 
    parser.add_argument("--final_model", required=True) 
    args = parser.parse_args()
    print("ARGS:", vars(args))
    if args.combined_model == "True":
        print("Training model for: COMBINED MODELS (MULTI-TASK FSL LEARNING)")
        TRAIN_DATASET, CONFIG = load_combined_model_dataset(model=args.model, final_model=args.final_model) 
        output_dir = CONFIG.model_output_path
        output_log_dir = CONFIG.output_log_dir
    elif args.final_model=="True":
        print("Training model for: FINAL FSL TRAINING FOR SUBMITING")
        CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model=args.model)
        CONFIG.subtask_train_train=CONFIG.subtask_train
        print("TRAIN File Path:", CONFIG.subtask_train_train)
        TEMPLATES = CONFIG.templates_dict['train']
        TRAIN_DATASET = load_train(CONFIG, TEMPLATES)
        output_dir = CONFIG.model_output_path+"-final"
        output_log_dir = CONFIG.model_output_path+"-final-log"
    else:
        print("Training model for: SINGLE TASK FSL LEARNING")
        CONFIG = BaseConfig().get_args(subtask=int(args.subtask), model=args.model)
        TEMPLATES = CONFIG.templates_dict['train']
        TRAIN_DATASET = load_train(CONFIG, TEMPLATES)
        output_dir = CONFIG.model_output_path_combined
        output_log_dir = CONFIG.model_output_path_combined+"-log"

    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.model_input_path)
    TOKENIZED_TRAIN_DATASET = TRAIN_DATASET.map(preprocess_function, batched=True, remove_columns=["text", "label"])
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.model_input_path, device_map='auto')

    DATA_COLLECTOR = DataCollatorForSeq2Seq(
        TOKENIZER,
        model=MODEL,
        label_pad_token_id=CONFIG.label_pad_token_id,
        pad_to_multiple_of=8
    )
    TRAINING_ARGS = Seq2SeqTrainingArguments(
        output_dir=output_log_dir,
        auto_find_batch_size=CONFIG.auto_find_batch_size,
        learning_rate=CONFIG.learning_rate,
        num_train_epochs=CONFIG.num_train_epochs,
        logging_dir=f"{output_log_dir}/logs",
        logging_strategy="steps",
        # load_best_model_at_end=True,
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard"
    )
    
    TRAINER = Seq2SeqTrainer(
        model=MODEL,
        args=TRAINING_ARGS,
        data_collator=DATA_COLLECTOR,
        train_dataset=TOKENIZED_TRAIN_DATASET["train"]
    )
    MODEL.config.use_cache = False  
    print("LOGS:", TRAINER.train())

    print("SAVING MODEL ..... ")
    TRAINER.save_model(output_dir)
    TOKENIZER.save_pretrained(output_dir)
    print("MODEL trained and saved into:", output_dir)

