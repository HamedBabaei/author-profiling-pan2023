import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')
nltk.download('punkt')

# Read the CSV file
df = pd.read_csv('combined_texts1.csv')

# Preprocess and remove stop words, URLs, hashtags, '@' symbols, 'RT', and non-alphanumeric characters from the text column
stop_words = set(stopwords.words('english'))
preprocessed_texts = []

for text in df['combined_text']:
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove hashtags, '@' symbols, and 'RT'
    text = re.sub(r'#\w+|\@\w+|RT|rt', '', text, flags=re.IGNORECASE)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stop words and convert to lowercase
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    
    # Join the filtered words back into a text
    preprocessed_text = ' '.join(filtered_words)
    
    # Append the preprocessed text to the list
    preprocessed_texts.append(preprocessed_text)

# Create a new DataFrame with the preprocessed texts
preprocessed_df = pd.DataFrame({'preprocessed_text': preprocessed_texts})

# Specify the file path to save the preprocessed texts
save_path = 'preprocessed_texts1.csv'

# Save the preprocessed DataFrame to a CSV file
preprocessed_df.to_csv(save_path, index=False)
