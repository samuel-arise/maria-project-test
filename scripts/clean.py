import os 
import re 
import pandas as pd 
import nltk 
from nltk.corpus import stopwords 
  
# Download required NLTK data (only needs to run once) 
nltk.download('stopwords', quiet=True) 
nltk.download('punkt', quiet=True) 
  
def clean_text(text): 
    if not isinstance(text, str): 
        return '' 
  
    # Remove URLs 
    text = re.sub(r'http\S+|www\.\S+', '', text) 
  
    # Remove HTML entities like &amp; &#39; 
    text = re.sub(r'&[a-z]+;|&#[0-9]+;', '', text) 
  
    # Remove emojis and special unicode characters 
    text = text.encode('ascii', 'ignore').decode('ascii') 
  
    # Remove special characters but keep basic punctuation 
    text = re.sub(r'[^a-zA-Z0-9 .,!?\'\-]', '', text) 
  
    # Remove extra whitespace 
    text = re.sub(r'\s+', ' ', text).strip() 
  
    return text 
  
  
def main(): 
    # Load raw data 
    df = pd.read_csv('data/raw/comments_raw.csv') 
    print(f'Raw comments loaded: {len(df)}') 
  
    # Remove duplicates 
    df = df.drop_duplicates(subset='text') 
    print(f'After removing duplicates: {len(df)}') 
  
    # Apply text cleaning 
    df['text_clean'] = df['text'].apply(clean_text) 
  
    # Remove empty comments after cleaning 
    df = df[df['text_clean'].str.len() > 10] 
    print(f'After removing empty/short comments: {len(df)}') 
  
    # Filter to English comments only 
    # (Basic filter — removes obvious non-English by character set) 
    df = df[df['text_clean'].str.contains('[a-zA-Z]', regex=True)] 
    print(f'After English filter: {len(df)}') 
  
    # Reset index 
    df = df.reset_index(drop=True) 
  
    # Save cleaned data 
    os.makedirs('data/cleaned', exist_ok=True) 
    df.to_csv('data/cleaned/comments_cleaned.csv', index=False) 
  
    print(f'\nCleaned dataset saved: data/cleaned/comments_cleaned.csv') 
    print(df[['outlet', 'text_clean']].head(10)) 
  
  
if __name__ == '__main__': 
    main()