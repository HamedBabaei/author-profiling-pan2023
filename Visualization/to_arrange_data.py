import pandas as pd

# Read the CSV file
df = pd.read_csv('subtask1_train_df.csv')

# Group texts by labels and combine them into lists
grouped_texts = df.groupby('label')['text'].apply(list).reset_index()

# Create a dictionary to store the combined texts for each label
combined_texts = {label: texts for label, texts in zip(grouped_texts['label'], grouped_texts['text'])}

# Create a new DataFrame from the combined texts
combined_df = pd.DataFrame({'label': list(combined_texts.keys()), 'combined_text': list(combined_texts.values())})

# Save the combined DataFrame to a CSV file
combined_df.to_csv('combined_texts1.csv', index=False)
