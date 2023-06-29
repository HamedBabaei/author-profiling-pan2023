import pandas as pd
import random
import json
import os
import re
import json
import codecs
import demoji

SEP_TOKEN = "<SEP>"

def save_json(data, path):
    """
        Write json file
    """
    with codecs.open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)

def load_truth(truth_path):
    """ 
        Load a labels.jsonl file, convert it to dictionary representation and return the dictionary.
    """
    label_dict = {}
    with open(truth_path, 'r') as inp:
        for i in inp:
            tmp=json.loads(i)
            label_dict[tmp['twitter user id']] = tmp['class']
    return label_dict

def load_text(text_path):
    """ 
         Load a train_text.json file, convert it to dataframe representation and return the dataframe.
    """
    user_id=[]
    user_text=[]
    with open(text_path, 'r') as inp:
        for i in inp:
            tmp=json.loads(i)
            user_id.append(tmp['twitter user id'])
            user_text.append(f"{SEP_TOKEN}".join([text['text'] for text in tmp['texts']]))
    df = pd.DataFrame(data={'twitter user id':user_id, 'text':user_text})
    return df

def cleaner(text):
    text = "  ".join(text.split(SEP_TOKEN))
    # emojis = demoji.findall(text)
    # for emoji, description in emojis.items():
    #     text = text.replace(emoji, " "+description)
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


def build_train_samples(tweets, labels, templates):
    x_train, y_train = [], []
    for tweet, label in zip(tweets, labels):
        cleaned_tweet = cleaner(tweet)
        for template in templates:
            x_train.append(template.replace("{tweets}", cleaned_tweet))
            y_train.append(label)
    return x_train, y_train

def cleaner_seprated(text):
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

def build_train_samples_seprated(tweets, labels, templates):
    x_train, y_train = [], []
    for tweet, label in zip(tweets, labels):
        seprated_tweets = tweet.split(SEP_TOKEN)
        for seprated_tweet in seprated_tweets:
            cleaned_tweet = cleaner_seprated(seprated_tweet)
            for template in templates:
                x_train.append(template.replace("{tweets}", cleaned_tweet))
                y_train.append(label)
    return x_train, y_train

def build_test_samples(tweets, templates):
    x_test = []
    for tweet in tweets:
        x_test.append(templates.replace("{tweets}", cleaner(tweet)))
    return x_test
