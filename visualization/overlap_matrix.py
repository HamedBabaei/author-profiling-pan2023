import pandas as pd

df = pd.read_csv("Subtask3/split_text_task3.csv")
tokens_dict = {}

for key, values in dict(df).items():
    tokens_dict[0] = list(set(eval(key)))
    for index, value in enumerate(values):
        tokens_dict[index+1] = list(set(eval(value)))

import numpy as np

freq_matrix = np.zeros((len(tokens_dict), len(tokens_dict)))
for row_index in range(freq_matrix.shape[0]):
    for col_index in range(freq_matrix.shape[1]):
        if row_index >= col_index:
            freq_matrix[row_index, col_index] = len(set(tokens_dict[row_index]).intersection(set(tokens_dict[col_index])))
print(freq_matrix)                    