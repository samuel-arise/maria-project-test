import json
import pandas as pd
import os

def main():
    # 1. Load the Apify JSON export
    # Prioritizing your specific filename (apify_x_data) and adding fallbacks
    file_paths_to_try = [
        'data/raw/apify_x_data.json',
        'data/raw/apify_x_data',       # In case it saved without the .json extension
        'data/raw/x_data.json',
        'data/raw/apify_tweets_raw.json'
    ]
    
    file_path = None
    for path in file_paths_to_try:
        if os.path.exists(path):
            file_path = path
            break
            
    if not file_path:
        print("Error: Could not find 'apify_x_data.json'. Ensure it is in the data/raw/ folder.")
        return

    print(f'Loading JSON file from {file_path}. Flattening data structure...')

    # 2. Load JSON data
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = json.load(f)

    # 3. Smart Flattening Function
    # This handles the "list inside a list" error
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
            # Extract location from author dictionary
            location = author_data.get('location') or ''
        else:
            author = str(author_data)
            location = ''

        # If location wasn't in author data, check root level just in case
        if not location:
            location = item.get('location') or item.get('user_location') or ''

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
                'user_location': location,  # <--- Added location extraction!
                'published_at': published_at,
                'likes':        likes,
                'reply_count':  reply_count,
                'source':       'X_Apify'
            })

    # 4. Convert to Pandas DataFrame
    df = pd.DataFrame(records)
    print(f'Valid records after filtering: {len(df)}')
    
    if len(df) > 0:
        print("\nFirst 3 rows of processed data (including location):")
        print(df[['outlet', 'author', 'user_location']].head(3))

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