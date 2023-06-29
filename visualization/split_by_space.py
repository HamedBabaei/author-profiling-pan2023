import pandas as pd

# Read the CSV file
df = pd.read_csv('preprocessed_texts1.csv')

# Split every row by space and save the resulting values
split_values = df['preprocessed_text'].str.split()

# Create a new DataFrame with the split values
split_df = pd.DataFrame({'split_text': split_values})

# Specify the file path to save the split DataFrame
save_path = 'split_text_task1.csv'

# Save the split DataFrame to a CSV file
split_df.to_csv(save_path, index=False)
