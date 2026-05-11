import os 
import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
  
def classify_sentiment(compound): 
    if compound >= 0.05: 
        return 'Positive' 
    elif compound <= -0.05: 
        return 'Negative' 
    else: 
        return 'Neutral' 
  
  
def main(): 
    # Load cleaned data 
    df = pd.read_csv('data/cleaned/comments_cleaned.csv') 
    print(f'Running sentiment analysis on {len(df)} comments...') 
  
    # Initialize VADER 
    analyzer = SentimentIntensityAnalyzer() 
  
    # Run sentiment analysis on each comment 
    results = df['text_clean'].apply(analyzer.polarity_scores) 
  
    # Extract individual scores into separate columns 
    df['score_positive'] = results.apply(lambda x: x['pos']) 
    df['score_negative'] = results.apply(lambda x: x['neg']) 
    df['score_neutral']  = results.apply(lambda x: x['neu']) 
    df['score_compound'] = results.apply(lambda x: x['compound']) 
  
    # Add sentiment label 
    df['sentiment_label'] = df['score_compound'].apply(classify_sentiment) 
  
    # Print summary 
    print('\n── Sentiment Summary ─────────────────────────') 
    print(df['sentiment_label'].value_counts()) 
    print('\n── By Outlet ─────────────────────────────────') 
    print(df.groupby(['outlet', 'sentiment_label']).size().unstack(fill_value=0)) 
    
# Save results 
    os.makedirs('outputs/exports', exist_ok=True) 
    df.to_csv('data/cleaned/comments_sentiment.csv', index=False) 
    df.to_excel('outputs/exports/sentiment_results.xlsx', index=False) 
    
    print('\nResults saved:') 
    print('  data/cleaned/comments_sentiment.csv') 
    print('  outputs/exports/sentiment_results.xlsx') 

if __name__ == '__main__': 
     main()