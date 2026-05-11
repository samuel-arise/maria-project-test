import os 
import time 
import pandas as pd 
from googleapiclient.discovery import build 
from dotenv import load_dotenv 
  
# Load your API key from the .env file 
load_dotenv() 
API_KEY = os.getenv('YOUTUBE_API_KEY') 
  
# Build the YouTube API connection 
youtube = build('youtube', 'v3', developerKey=API_KEY) 
  
#video IDs here ────────────────────────────────── 
VIDEOS = { 
    'BBC':        'nK7-DW4oG58', 
    'Al_Jazeera': 'HfdLO3QwjtA', 
    'CHANNELS':   '8mR-_5r3lJM', 
    'TVC':        'L9ZGqeUas-k',
} 
  
def get_comments(video_id, outlet_name, max_results=500): 
    comments = [] 
    next_page_token = None 
  
    print(f'Collecting comments from {outlet_name}...') 
  
    while True: 
        request = youtube.commentThreads().list( 
            part='snippet', 
            videoId=video_id, 
            maxResults=100, 
            pageToken=next_page_token, 
            textFormat='plainText', 
            order='relevance' 
        ) 
        response = request.execute() 
  
        for item in response['items']: 
            top = item['snippet']['topLevelComment']['snippet'] 
            comments.append({ 
                'outlet':       outlet_name, 
                'video_id':     video_id, 
                'author':       top['authorDisplayName'], 
                'text':         top['textDisplay'], 
                'likes':        top['likeCount'], 
                'published_at': top['publishedAt'], 
                'reply_count':  item['snippet']['totalReplyCount'] 
            }) 
  
        next_page_token = response.get('nextPageToken') 
  
        if not next_page_token or len(comments) >= max_results: 
            break 
  
        # Pause briefly to respect YouTube's rate limits 
        time.sleep(0.5) 
  
    print(f'  Done. Collected {len(comments)} comments from {outlet_name}.') 
    return comments 
  
  
def main(): 
    all_comments = [] 
  
    for outlet, video_id in VIDEOS.items(): 
        comments = get_comments(video_id, outlet) 
        all_comments.extend(comments) 
  
    df = pd.DataFrame(all_comments) 
  
    # Save raw data — never edit this file manually 
    os.makedirs('data/raw', exist_ok=True) 
    df.to_csv('data/raw/comments_raw.csv', index=False) 
  
    print(f'\nTotal comments collected: {len(df)}') 
    print('Saved to: data/raw/comments_raw.csv') 
    print(df.head()) 
  
  
if __name__ == '__main__': 
    main()