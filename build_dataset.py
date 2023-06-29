from utils import load_truth, load_text
from config import BaseConfig
import pandas


if __name__ == "__main__":
    
    for subtask in range(1,4):
        config = BaseConfig().get_args(subtask=subtask, model="None")

        df = load_text(config.subtask_text)
        labels_dict = load_truth(config.subtask_truth)
        labels = []
        for user_id in df['twitter user id'].tolist():
            labels.append(labels_dict[user_id])
        df['label'] = labels

        print(df['label'].value_counts())
        df.to_csv(config.subtask_train)
        
        gp = df.groupby("label")
        train, test = [], []
        for index, gp_df in gp:
            train_size = 32 - config.test_size_subtask1 if subtask == 1 else 64 - config.test_size_subtask23
            test_size = config.test_size_subtask1 if subtask == 1 else config.test_size_subtask23
            train.append(gp_df.head(train_size))
            test.append(gp_df.tail(test_size))

        train_df = pandas.concat(train).reset_index(drop=True)
        test_df = pandas.concat(test).reset_index(drop=True)

        print(train_df['label'].value_counts())
        print(test_df['label'].value_counts())
          
        train_df.to_csv(config.subtask_train_train)
        test_df.to_csv(config.subtask_train_test)
