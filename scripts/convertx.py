import json
import pandas as pd
import os

def main():
    # 1. Load the Apify JSON export
    # We will check for both common naming conventions just in case
    file_path = 'data/raw/apify_tweets_raw.json'
    if not os.path.exists(file_path):
        file_path = 'data/raw/apify_x_data.json'
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the Apify JSON file. Ensure it is in the data/raw/ folder.")
        return

    print('Loaded JSON file. Flattening data structure...')

    # 2. Smart Flattening Function
    # This handles the "list inside a list" error you experienced
    flat_records = []
    def flatten_data(item):
        if isinstance(item, list):
            for i in item:
                flatten_data(i)
        elif isinstance(item, dict):
            flat_records.append(item)

    flatten_data(raw)
    print(f'Found {len(flat_records)} total tweet dictionaries.')

    # 3. Map the columns safely
    records = []
    for item in flat_records:
        # Safely grab the text
        text = item.get('full_text') or item.get('text') or item.get('rawContent', '')
        
        # Safely grab the author
        author_data = item.get('author') or item.get('user') or {}
        if isinstance(author_data, dict):
            author = author_data.get('userName') or author_data.get('name') or author_data.get('screen_name') or 'Unknown'
        else:
            author = str(author_data)

        # Safely grab metrics
        published_at = item.get('createdAt') or item.get('created_at', '')
        likes = item.get('likeCount') or item.get('favorite_count', 0)
        reply_count = item.get('replyCount') or item.get('reply_count', 0)

        # Only append if there is actual text
        if str(text).strip():
            records.append({
                'outlet':       classify_outlet(text),
                'text':         text,
                'author':       author,
                'published_at': published_at,
                'likes':        likes,
                'reply_count':  reply_count,
                'source':       'X_Apify'
            })

    # 4. Convert to Pandas DataFrame
    df = pd.DataFrame(records)
    print(f'Valid records after filtering: {len(df)}')
    
    if len(df) > 0:
        print("\nFirst 3 rows of processed data:")
        print(df[['outlet', 'text', 'author']].head(3))

    # 5. Save the output
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/x_apify_cleaned.csv', index=False)
    print('\nSUCCESS: Saved data/raw/x_apify_cleaned.csv')
    print('You can now run: python scripts/merge.py')

def classify_outlet(text):
    # Attempt to classify which outlet the tweet relates to
    text_lower = str(text).lower()
    if 'bbc' in text_lower:
        return 'BBC'
    if 'aljazeera' in text_lower or 'al jazeera' in text_lower:
        return 'Al_Jazeera'
    return 'X_General'

if __name__ == '__main__':
    main()