import os
import time
import pandas as pd
import pycountry
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from dotenv import load_dotenv
 
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=API_KEY)
 
 
def get_channel_country(channel_id):
    try:
        request = youtube.channels().list(
            part='snippet',
            id=channel_id
        )
        response = request.execute()
        items = response.get('items', [])
        if items:
            country_code = items[0]['snippet'].get('country', None)
            return country_code
        return None
    except Exception as e:
        print(f'  Error fetching channel {channel_id}: {e}')
        return None
 
 
def code_to_country_name(code):
    if not code:
        return 'Not Specified'
    try:
        country = pycountry.countries.get(alpha_2=code.upper())
        return country.name if country else code
    except:
        return code
 
 
def code_to_region(code):
    # Maps country codes to continental regions
    africa = ['NG','GH','KE','ZA','ET','TZ','UG','RW','SN','CI',
              'CM','CD','AO','MZ','ZM','ZW','BW','NA','MW','MG',
              'SO','SD','ER','DJ','LY','TN','DZ','MA','EG','ML',
              'BF','NE','TD','MR','GM','SL','LR','GN','TG','BJ',
              'GA','GQ','CG','CF','BI','RW','SS','LS','SZ'],
    west_africa = ['NG','GH','SN','CI','CM','ML','BF','NE','GM',
                   'SL','LR','GN','TG','BJ','MR'],
    east_africa = ['KE','TZ','UG','ET','RW','BI','SO','DJ','ER','SS'],
    north_america = ['US','CA','MX'],
    europe = ['GB','DE','FR','IT','ES','NL','BE','PL','SE','NO',
              'DK','FI','AT','CH','PT','GR','CZ','SK','HU','RO',
              'BG','HR','RS','UA','IE','IS','LU','MT','CY','EE',
              'LV','LT','SI','AL','MK','BA','ME','XK'],
    middle_east = ['SA','AE','QA','KW','BH','OM','YE','IQ','IR',
                   'JO','LB','SY','IL','PS','TR'],
    asia = ['IN','PK','BD','LK','NP','MM','TH','VN','ID','MY',
            'PH','SG','KH','LA','CN','JP','KR','TW','HK','MO',
            'MN','KZ','UZ','TM','TJ','KG','AF'],
    latin_america = ['BR','AR','CL','CO','PE','VE','EC','BO','PY',
                     'UY','GY','SR','CU','DO','PR','HT','JM','TT',
                     'BB','GD','LC','VC','AG','DM','KN','BS'],
    oceania = ['AU','NZ','FJ','PG','SB','VU','WS','TO','KI','FM']
 
    if not code:
        return 'Unknown'
    code = code.upper()
    if code in ['NG','GH','SN','CI','CM','ML','BF','NE','GM','SL','LR','GN','TG','BJ','MR']:
        return 'West Africa'
    if code in ['KE','TZ','UG','ET','RW','BI','SO','DJ','ER','SS']:
        return 'East Africa'
    if code in ['ZA','BW','NA','ZW','ZM','MW','LS','SZ','MZ']:
        return 'Southern Africa'
    if code in ['EG','LY','TN','DZ','MA','SD']:
        return 'North Africa'
    if code in ['US','CA','MX']:
        return 'North America'
    if code in ['BR','AR','CL','CO','PE','VE','EC','BO','PY','UY','GY','SR','CU','DO','JM','TT']:
        return 'Latin America'
    if code in ['GB','DE','FR','IT','ES','NL','BE','PL','SE','NO','DK','FI','AT','CH','PT','GR','IE','CZ','SK','HU','RO','BG','HR']:
        return 'Europe'
    if code in ['SA','AE','QA','KW','BH','OM','YE','IQ','IR','JO','LB','SY','IL','PS','TR']:
        return 'Middle East'
    if code in ['IN','PK','BD','LK','NP','MM','TH','VN','ID','MY','PH','SG','CN','JP','KR','TW']:
        return 'Asia'
    if code in ['AU','NZ','FJ','PG']:
        return 'Oceania'
    return 'Other'
 
 
def main():
    # Load the scraped data which contains author channel IDs
    raw_df = pd.read_csv('data/raw/comments_raw.csv')
    cleaned_df = pd.read_csv('data/cleaned/comments_cleaned.csv')
 
    # Check if author column exists
    if 'author' not in raw_df.columns:
        print('ERROR: author column not found in raw data.')
        print('Available columns:', raw_df.columns.tolist())
        return
 
    # Get unique authors to minimise API calls
    unique_authors = raw_df['author'].dropna().unique()
    print(f'Looking up countries for {len(unique_authors)} unique commenters...')
    print('This may take several minutes depending on dataset size.')
 
    # Build author -> country mapping
    author_country_map = {}
    for i, author in enumerate(unique_authors):
        country_code = get_channel_country(author)
        author_country_map[author] = country_code
 
        # Progress update every 50 lookups
        if (i + 1) % 50 == 0:
            found = sum(1 for v in author_country_map.values() if v)
            print(f'  Progress: {i+1}/{len(unique_authors)} | Countries found: {found}')
 
        # Small delay to respect API rate limits
        time.sleep(0.3)
 
    # Map countries back to the cleaned dataframe
    raw_df['country_code'] = raw_df['author'].map(author_country_map)
 
    # Add country name and region
    raw_df['country_name'] = raw_df['country_code'].apply(code_to_country_name)
    raw_df['region'] = raw_df['country_code'].apply(code_to_region)
 
    # Merge with cleaned data
    if 'country_code' not in cleaned_df.columns:
        geo_cols = raw_df[['author','country_code','country_name','region']].copy()
        # Note: merge on author display name (not perfect but workable)
        cleaned_df = cleaned_df.merge(
            geo_cols.drop_duplicates('author'),
            on='author',
            how='left'
        )
 
    # ── Summary Statistics ────────────────────────────────────
    total = len(cleaned_df)
    with_country = cleaned_df['country_code'].notna().sum()
    coverage = round(with_country / total * 100, 1)
 
    print(f'\n── Geographic Coverage ───────────────────────────────')
    print(f'Total comments:        {total}')
    print(f'With country data:     {with_country} ({coverage}%)')
    print(f'Without country data:  {total - with_country}')
 
    print(f'\n── Top Countries ─────────────────────────────────────')
    country_counts = cleaned_df['country_name'].value_counts()
    print(country_counts[country_counts.index != 'Not Specified'].head(15))
 
    print(f'\n── Regional Distribution ─────────────────────────────')
    region_counts = cleaned_df['region'].value_counts()
    print(region_counts)
 
    # ── Save Results ──────────────────────────────────────────
    cleaned_df.to_csv('data/cleaned/comments_with_geo.csv', index=False)
    print('\nSaved: data/cleaned/comments_with_geo.csv')
 
    # ── Chart 1: Regional Distribution Pie Chart ──────────────
    os.makedirs('outputs/charts', exist_ok=True)
    region_data = cleaned_df[cleaned_df['region'] != 'Unknown']['region'].value_counts()
 
    colors = [
        '#2E86AB','#27AE60','#E74C3C','#F39C12','#8E44AD',
        '#1ABC9C','#E67E22','#2C3E50','#D35400','#7F8C8D'
    ]
 
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        region_data.values,
        labels=region_data.index,
        colors=colors[:len(region_data)],
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 11}
    )
    ax.set_title(
        'Regional Distribution of Commenters\n(Based on YouTube Channel Country Setting)',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig('outputs/charts/07_regional_distribution.png', dpi=150)
    plt.close()
    print('Chart saved: outputs/charts/07_regional_distribution.png')
 
    # ── Chart 2: Top 15 Countries Bar Chart ───────────────────
    top_countries = country_counts[
        country_counts.index != 'Not Specified'
    ].head(15)
 
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        range(len(top_countries)),
        top_countries.values,
        color='#2E86AB',
        edgecolor='white'
    )
    ax.set_yticks(range(len(top_countries)))
    ax.set_yticklabels(top_countries.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(
        'Top 15 Countries — Verified Commenter Locations\nNatasha-Akpabio Saga Coverage',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Number of Commenters', fontsize=11)
 
    for bar, count in zip(bars, top_countries.values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va='center', fontsize=9, fontweight='bold'
        )
 
    plt.tight_layout()
    plt.savefig('outputs/charts/08_top_countries.png', dpi=150)
    plt.close()
    print('Chart saved: outputs/charts/08_top_countries.png')
 
    # ── Chart 3: Sentiment by Region ──────────────────────────
    sentiment_df = pd.read_csv('data/cleaned/comments_sentiment.csv')
    geo_df = cleaned_df[['author','region']].drop_duplicates('author')
    merged = sentiment_df.merge(geo_df, on='author', how='left')
    merged = merged[merged['region'].notna() & (merged['region'] != 'Unknown')]
 
    if len(merged) > 0:
        pivot = merged.groupby(
            ['region','sentiment_label']
        ).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
 
        ax = pivot_pct.plot(
            kind='bar',
            color=['#E74C3C','#F39C12','#27AE60'],
            figsize=(12, 6),
            edgecolor='white',
            width=0.7
        )
        ax.set_title(
            'Sentiment Distribution by World Region\n(% of comments per region)',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('World Region', fontsize=11)
        ax.set_ylabel('Percentage of Comments (%)', fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig('outputs/charts/09_sentiment_by_region.png', dpi=150)
        plt.close()
        print('Chart saved: outputs/charts/09_sentiment_by_region.png')
 
    print('\nGeographic analysis complete.')
 
 
if __name__ == '__main__':
    main()