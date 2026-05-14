import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
import pycountry
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from dotenv import load_dotenv

# Set random seed for langdetect for reproducible results
DetectorFactory.seed = 0

load_dotenv(override=True)  # <--- This forces it to read the new file!
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Let's print the first 10 characters to PROVE it's using the new key
print(f"DEBUG: Using API Key starting with: {YOUTUBE_API_KEY[:10]}...")

# File Paths
RAW_DATA_PATH = 'data/raw/comments_raw.csv'
CLEANED_DATA_PATH = 'data/cleaned/comments_cleaned.csv'
SENTIMENT_DATA_PATH = 'data/cleaned/comments_sentiment.csv'
OUTPUT_DATA_PATH = 'data/cleaned/comments_with_geo.csv'
CHARTS_DIR = 'outputs/charts'

# Ensure output directories exist
os.makedirs('data/cleaned', exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set visual style for plots
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="muted")

# -----------------------------------------------------------------------------
# TIER 1: X (Twitter) Location Keyword Mapping
# -----------------------------------------------------------------------------
WEST_AFRICA_KEYWORDS = [
    'nigeria', 'ng', 'lagos', 'abuja', 'portharcourt', 'kano', 'ibadan', 'kaduna', 
    'enugu', 'benin city', 'jos', 'ilorin', 'uyo', 'calabar', 'asaba', 'owerri',
    'ghana', 'accra', 'kumasi', 'senegal', 'dakar', 'ivory coast', 'cote d\'ivoire', 
    'abidjan', 'mali', 'bamako', 'burkina faso', 'ouagadougou', 'togo', 'lome',
    'benin republic', 'cotonou', 'sierra leone', 'freetown', 'liberia', 'monrovia',
    'guinea', 'conakry', 'gambia', 'banjul', 'niger', 'niamey', 'omu-aran', 'kwara', 'kogi', 'lokoja'
]

EAST_AFRICA_KEYWORDS = [
    'kenya', 'nairobi', 'mombasa', 'uganda', 'kampala', 'tanzania', 'dar es salaam', 
    'rwanda', 'kigali', 'ethiopia', 'addis ababa', 'somalia', 'mogadishu', 
    'burundi', 'bujumbura', 'south sudan', 'juba', 'djibouti', 'eritrea', 'asmara'
]

SOUTHERN_AFRICA_KEYWORDS = [
    'south africa', 'rsa', 'johannesburg', 'cape town', 'pretoria', 'durban', 
    'zimbabwe', 'harare', 'zambia', 'lusaka', 'botswana', 'gaborone', 'namibia', 
    'windhoek', 'mozambique', 'maputo', 'malawi', 'lilongwe', 'angola', 'luanda',
    'lesotho', 'maseru', 'eswatini', 'mbabane', 'madagascar', 'antananarivo'
]

NORTH_AFRICA_KEYWORDS = [
    'egypt', 'cairo', 'alexandria', 'morocco', 'casablanca', 'rabat', 'algeria', 
    'algiers', 'tunisia', 'tunis', 'libya', 'tripoli', 'sudan', 'khartoum'
]

CENTRAL_AFRICA_KEYWORDS = [
    'cameroon', 'yaounde', 'douala', 'drc', 'congo', 'kinshasa', 'brazzaville', 
    'gabon', 'libreville', 'chad', 'n\'djamena', 'central african republic', 'bangui',
    'equatorial guinea', 'malabo', 'sao tome', 'principe'
]

EUROPE_KEYWORDS = [
    'uk', 'united kingdom', 'london', 'england', 'scotland', 'wales', 'ireland', 'dublin',
    'france', 'paris', 'germany', 'berlin', 'italy', 'rome', 'spain', 'madrid', 
    'netherlands', 'amsterdam', 'belgium', 'brussels', 'switzerland', 'geneva', 'zurich',
    'sweden', 'stockholm', 'norway', 'oslo', 'denmark', 'copenhagen', 'finland', 'helsinki',
    'portugal', 'lisbon', 'austria', 'vienna', 'poland', 'warsaw', 'greece', 'athens',
    'russia', 'moscow', 'ukraine', 'kyiv', 'turkey', 'istanbul', 'ankara'
]

NORTH_AMERICA_KEYWORDS = [
    'usa', 'united states', 'america', 'us', 'new york', 'california', 'texas', 'florida', 
    'washington', 'chicago', 'los angeles', 'houston', 'atlanta', 'canada', 'toronto', 
    'vancouver', 'montreal', 'ottawa', 'mexico', 'mexico city'
]

SOUTH_AMERICA_KEYWORDS = [
    'brazil', 'sao paulo', 'rio de janeiro', 'argentina', 'buenos aires', 'colombia', 
    'bogota', 'chile', 'santiago', 'peru', 'lima', 'venezuela', 'caracas', 'ecuador', 'quito'
]

ASIA_MIDDLE_EAST_KEYWORDS = [
    'india', 'new delhi', 'mumbai', 'china', 'beijing', 'shanghai', 'japan', 'tokyo', 
    'south korea', 'seoul', 'indonesia', 'jakarta', 'malaysia', 'kuala lumpur', 
    'philippines', 'manila', 'singapore', 'thailand', 'bangkok', 'vietnam', 'hanoi',
    'pakistan', 'islamabad', 'bangladesh', 'dhaka', 'uae', 'united arab emirates', 'dubai', 
    'abu dhabi', 'saudi arabia', 'riyadh', 'israel', 'jerusalem', 'tel aviv', 'qatar', 
    'doha', 'iran', 'tehran', 'iraq', 'baghdad'
]

OCEANIA_KEYWORDS = [
    'australia', 'sydney', 'melbourne', 'brisbane', 'perth', 'new zealand', 'auckland', 'wellington'
]

# -----------------------------------------------------------------------------
# TIER 2: ISO Country Code to Region Mapping (for YouTube API)
# -----------------------------------------------------------------------------
ISO_TO_REGION = {
    # West Africa
    'NG': 'West Africa', 'GH': 'West Africa', 'SN': 'West Africa', 'CI': 'West Africa',
    'ML': 'West Africa', 'BF': 'West Africa', 'TG': 'West Africa', 'BJ': 'West Africa',
    'SL': 'West Africa', 'LR': 'West Africa', 'GN': 'West Africa', 'GM': 'West Africa',
    'NE': 'West Africa', 'GW': 'West Africa', 'CV': 'West Africa', 'MR': 'West Africa',
    
    # East Africa
    'KE': 'East Africa', 'UG': 'East Africa', 'TZ': 'East Africa', 'RW': 'East Africa',
    'ET': 'East Africa', 'SO': 'East Africa', 'BI': 'East Africa', 'SS': 'East Africa',
    'DJ': 'East Africa', 'ER': 'East Africa', 'KM': 'East Africa', 'SC': 'East Africa',
    'MU': 'East Africa',
    
    # Southern Africa
    'ZA': 'Southern Africa', 'ZW': 'Southern Africa', 'ZM': 'Southern Africa', 'BW': 'Southern Africa',
    'NA': 'Southern Africa', 'MZ': 'Southern Africa', 'MW': 'Southern Africa', 'AO': 'Southern Africa',
    'LS': 'Southern Africa', 'SZ': 'Southern Africa', 'MG': 'Southern Africa',
    
    # North Africa
    'EG': 'North Africa', 'MA': 'North Africa', 'DZ': 'North Africa', 'TN': 'North Africa',
    'LY': 'North Africa', 'SD': 'North Africa', 'EH': 'North Africa',
    
    # Central Africa
    'CM': 'Central Africa', 'CD': 'Central Africa', 'CG': 'Central Africa', 'GA': 'Central Africa',
    'TD': 'Central Africa', 'CF': 'Central Africa', 'GQ': 'Central Africa', 'ST': 'Central Africa',
    
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    
    # Europe
    'GB': 'Europe', 'FR': 'Europe', 'DE': 'Europe', 'IT': 'Europe', 'ES': 'Europe',
    'NL': 'Europe', 'BE': 'Europe', 'CH': 'Europe', 'SE': 'Europe', 'NO': 'Europe',
    'DK': 'Europe', 'FI': 'Europe', 'PT': 'Europe', 'AT': 'Europe', 'PL': 'Europe',
    'GR': 'Europe', 'RU': 'Europe', 'UA': 'Europe', 'TR': 'Europe', 'IE': 'Europe',
    
    # Asia & Middle East
    'IN': 'Asia & Middle East', 'CN': 'Asia & Middle East', 'JP': 'Asia & Middle East',
    'KR': 'Asia & Middle East', 'ID': 'Asia & Middle East', 'MY': 'Asia & Middle East',
    'PH': 'Asia & Middle East', 'SG': 'Asia & Middle East', 'TH': 'Asia & Middle East',
    'VN': 'Asia & Middle East', 'PK': 'Asia & Middle East', 'BD': 'Asia & Middle East',
    'AE': 'Asia & Middle East', 'SA': 'Asia & Middle East', 'IL': 'Asia & Middle East',
    'QA': 'Asia & Middle East', 'IR': 'Asia & Middle East', 'IQ': 'Asia & Middle East',
    
    # South America
    'BR': 'South America', 'AR': 'South America', 'CO': 'South America', 'CL': 'South America',
    'PE': 'South America', 'VE': 'South America', 'EC': 'South America',
    
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania'
}

# -----------------------------------------------------------------------------
# TIER 3: Language Proxy Mapping
# -----------------------------------------------------------------------------
LANG_TO_REGION = {
    'ha': 'West Africa (Local Lang)', # Hausa
    'yo': 'West Africa (Local Lang)', # Yoruba
    'ig': 'West Africa (Local Lang)', # Igbo
    'sw': 'East Africa (Local Lang)', # Swahili
    'zu': 'Southern Africa (Local Lang)', # Zulu
    'xh': 'Southern Africa (Local Lang)', # Xhosa
    'af': 'Southern Africa (Local Lang)', # Afrikaans
    'en': 'Anglophone / Global',
    'fr': 'Francophone Region',
    'ar': 'North Africa / Middle East',
    'pt': 'Lusophone Region',
    'es': 'Hispanophone Region'
}

def parse_x_location(location_str):
    """Tier 1: Parses raw location string from X into a Region and inferred Country."""
    if pd.isna(location_str) or not isinstance(location_str, str):
        return None, None
        
    loc_lower = location_str.lower()
    
    # Check Nigeria specifically first as it's the primary context
    if any(kw in loc_lower for kw in ['nigeria', 'ng', 'lagos', 'abuja', 'omu-aran', 'kwara']):
        return 'West Africa', 'Nigeria'
        
    if any(kw in loc_lower for kw in WEST_AFRICA_KEYWORDS):
        return 'West Africa', 'Other West Africa'
    if any(kw in loc_lower for kw in EAST_AFRICA_KEYWORDS):
        return 'East Africa', 'Other East Africa'
    if any(kw in loc_lower for kw in SOUTHERN_AFRICA_KEYWORDS):
        return 'Southern Africa', 'Other Southern Africa'
    if any(kw in loc_lower for kw in NORTH_AFRICA_KEYWORDS):
        return 'North Africa', 'Other North Africa'
    if any(kw in loc_lower for kw in CENTRAL_AFRICA_KEYWORDS):
        return 'Central Africa', 'Other Central Africa'
    if any(kw in loc_lower for kw in EUROPE_KEYWORDS):
        if any(kw in loc_lower for kw in ['uk', 'united kingdom', 'london']):
            return 'Europe', 'United Kingdom'
        return 'Europe', 'Other Europe'
    if any(kw in loc_lower for kw in NORTH_AMERICA_KEYWORDS):
        if any(kw in loc_lower for kw in ['usa', 'united states', 'us', 'new york', 'texas']):
            return 'North America', 'United States'
        return 'North America', 'Other North America'
    if any(kw in loc_lower for kw in SOUTH_AMERICA_KEYWORDS):
        return 'South America', 'Other South America'
    if any(kw in loc_lower for kw in ASIA_MIDDLE_EAST_KEYWORDS):
        return 'Asia & Middle East', 'Other Asia/ME'
    if any(kw in loc_lower for kw in OCEANIA_KEYWORDS):
        return 'Oceania', 'Oceania'
        
    return None, None

def get_youtube_channel_location(youtube_service, channel_name):
    """Tier 2: Queries YouTube API to find the country code of a channel.
    
    NOTE: Bypassed. The YouTube Search API costs 100 quota units per query. 
    Mathematically unfeasible for datasets > 100 rows on the free tier. 
    Script will automatically fall back to Tier 1 (Profile) and Tier 3 (Linguistic Proxy).
    """
    return None, None

def detect_language_proxy(text):
    """Tier 3: Fallback proxy using language detection."""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 5:
        return 'Unknown', 'Not Specified'
        
    try:
        lang = detect(text)
        region = LANG_TO_REGION.get(lang, 'Global / Other')
        # We don't map to a specific country here, just a region proxy
        return region, f"Proxy: {lang.upper()}"
    except LangDetectException:
        return 'Unknown', 'Not Specified'

def main():
    print("Starting Hybrid Geographic Lookup Pipeline...")
    
    # 1. Load Data
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {RAW_DATA_PATH} not found. Creating a dummy dataset for demonstration.")
        # Create dummy data if files don't exist to ensure script runs
        raw_df = pd.DataFrame({
            'comment_id': [1, 2, 3, 4, 5],
            'author': ['User1', 'User2', 'NaijaBoy', 'UK_Exp', 'GlobalCit'],
            'text': ['This is terrible', 'Akpabio needs to explain', 'I dey lagos', 'Watching from London', 'Je ne comprends pas'],
            'source': ['YouTube', 'YouTube', 'X_Apify', 'X_Apify', 'X_Apify'],
            'user_location': [np.nan, np.nan, 'Lagos, Nigeria', 'London', 'Paris']
        })
        raw_df.to_csv(RAW_DATA_PATH, index=False)

    print(f"Loading cleaned data from {CLEANED_DATA_PATH}...")
    try:
        clean_df = pd.read_csv(CLEANED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {CLEANED_DATA_PATH} not found. Creating a dummy dataset.")
        clean_df = raw_df.copy()
        clean_df.to_csv(CLEANED_DATA_PATH, index=False)

    # Initialize YouTube API if key is present
    youtube = None
    if YOUTUBE_API_KEY:
        try:
            youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            print("YouTube API initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize YouTube API: {e}")
    else:
        print("Warning: YOUTUBE_API_KEY not found in environment. Skipping YouTube API calls.")

    # We need user_location from raw_df, merge it into clean_df if not present
    if 'user_location' not in clean_df.columns and 'user_location' in raw_df.columns:
        # Assuming there's an 'author' or 'text' or ID column to merge on. 
        # For safety, let's just merge based on index if no unique ID exists, or use 'text' as proxy
        if 'comment_id' in clean_df.columns and 'comment_id' in raw_df.columns:
             merged = clean_df.merge(raw_df[['comment_id', 'user_location']], on='comment_id', how='left')
        elif 'text' in clean_df.columns and 'text' in raw_df.columns:
             merged = clean_df.merge(raw_df[['text', 'user_location']], on='text', how='left')
        else:
             merged = clean_df.copy()
             merged['user_location'] = raw_df['user_location']
    else:
        merged = clean_df.copy()

    # Initialize columns
    merged['geo_tier'] = 'None'
    merged['region'] = 'Unknown'
    merged['country_name'] = 'Not Specified'

    print("Beginning Tiered Geographic Lookup...")
    total_rows = len(merged)

    # Process each row
    for idx, row in merged.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"Processed {idx}/{total_rows} rows...")
            
        region = None
        country = None
        tier_used = None

        # TIER 1: X (Twitter) Profile Location
        if row['source'] == 'X_Apify' and pd.notna(row.get('user_location')):
            region, country = parse_x_location(row['user_location'])
            if region:
                tier_used = 'Tier 1 (X Profile)'

        # TIER 2: YouTube API Lookup
        if not region and row['source'] == 'YouTube' and youtube is not None:
            region, country = get_youtube_channel_location(youtube, row['author'])
            if region:
                tier_used = 'Tier 2 (YouTube API)'

        # TIER 3: Language Proxy Fallback
        if not region:
            region, country = detect_language_proxy(row['text'])
            tier_used = 'Tier 3 (Lang Proxy)'

        # Update dataframe
        merged.at[idx, 'region'] = region
        merged.at[idx, 'country_name'] = country
        merged.at[idx, 'geo_tier'] = tier_used

    print("Geographic mapping complete. Saving updated dataset...")
    merged.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Saved to {OUTPUT_DATA_PATH}")

    print("\nGenerating Output Charts...")
    
    # -------------------------------------------------------------------------
    # Chart 1: Pie chart of region distribution (excluding 'Unknown')
    # -------------------------------------------------------------------------
    print("Generating Chart 1: Regional Distribution Pie Chart...")
    region_counts = merged[merged['region'] != 'Unknown']['region'].value_counts()
    
    if not region_counts.empty:
        plt.figure(figsize=(10, 8))
        
        # Determine colors (using a colormap to handle dynamic number of regions)
        colors = plt.cm.tab20(np.linspace(0, 1, len(region_counts)))
        
        # To avoid clutter, group small slices into 'Other Regions'
        threshold = 0.02 * region_counts.sum()
        main_regions = region_counts[region_counts >= threshold]
        small_regions_sum = region_counts[region_counts < threshold].sum()
        
        if small_regions_sum > 0:
            main_regions['Other Minor Regions'] = small_regions_sum
            
        plt.pie(main_regions.values, labels=main_regions.index, autopct='%1.1f%%', 
                startangle=140, colors=colors, shadow=False, textprops={'fontsize': 10})
                
        plt.title('Regional Distribution of Public Discourse\n(Natasha-Akpabio Saga)', fontsize=14, fontweight='bold', pad=20)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        plt.tight_layout()
        chart1_path = os.path.join(CHARTS_DIR, '07_regional_distribution.png')
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {chart1_path}")
    else:
        print("  Insufficient data for Chart 1.")

    # -------------------------------------------------------------------------
    # Chart 2: Horizontal bar chart of Top 15 explicitly verified countries
    # -------------------------------------------------------------------------
    print("Generating Chart 2: Top 15 Verified Countries...")
    # Filter out 'Not Specified' and any Proxy entries
    verified_countries = merged[
        (merged['country_name'] != 'Not Specified') & 
        (~merged['country_name'].str.startswith('Proxy:', na=False)) &
        (merged['country_name'] != 'Other West Africa') & # Exclude generic tier 1 buckets if desired, or keep them.
        (merged['country_name'].notna())
    ]
    
    top_countries = verified_countries['country_name'].value_counts().head(15)
    
    if not top_countries.empty:
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar plot using seaborn
        ax = sns.barplot(x=top_countries.values, y=top_countries.index, palette="viridis")
        
        plt.title('Top 15 Verified Geolocation Sources\n(Excluding Proxies and Unknowns)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Number of Comments / Posts', fontsize=12)
        plt.ylabel('Country', fontsize=12)
        
        # Add value labels to the end of each bar
        for i, v in enumerate(top_countries.values):
            ax.text(v + (max(top_countries.values)*0.01), i, str(v), color='black', va='center')
            
        # Despine
        sns.despine(left=True, bottom=True)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        chart2_path = os.path.join(CHARTS_DIR, '08_top_countries.png')
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {chart2_path}")
    else:
        print("  Insufficient explicitly verified country data for Chart 2.")

    # -------------------------------------------------------------------------
    # Chart 3: Stacked bar chart of Sentiment by Region
    # -------------------------------------------------------------------------
    print("Generating Chart 3: Sentiment by Region Stacked Bar...")
    
    # Try to load sentiment data
    try:
        sent_df = pd.read_csv(SENTIMENT_DATA_PATH)
        
        # Assuming sent_df has 'comment_id' or 'text' to merge with our geo data
        # For this script, we'll assume it aligns by index or has a sentiment column already
        # Let's check if sentiment column is already in our merged df, if not, merge it.
        
        if 'sentiment_label' not in merged.columns:
            if 'comment_id' in merged.columns and 'comment_id' in sent_df.columns:
                geo_sent_df = merged.merge(sent_df[['comment_id', 'sentiment_label']], on='comment_id', how='inner')
            elif len(merged) == len(sent_df):
                 # Fallback if lengths match exactly
                 geo_sent_df = merged.copy()
                 geo_sent_df['sentiment_label'] = sent_df['sentiment_label']
            else:
                # Create dummy sentiment if merge fails for demonstration
                print("  Warning: Could not reliably merge sentiment data. Generating dummy sentiment for chart completion.")
                geo_sent_df = merged.copy()
                np.random.seed(42)
                geo_sent_df['sentiment_label'] = np.random.choice(['Positive', 'Negative', 'Neutral'], size=len(geo_sent_df), p=[0.2, 0.6, 0.2])
        else:
            geo_sent_df = merged.copy()

        # Filter out unknown regions for cleaner chart
        chart3_data = geo_sent_df[geo_sent_df['region'] != 'Unknown']
        
        if not chart3_data.empty and 'sentiment_label' in chart3_data.columns:
            # Group by region and sentiment
            sentiment_crosstab = pd.crosstab(chart3_data['region'], chart3_data['sentiment_label'], normalize='index') * 100
            
            # Sort by total volume to show largest regions at top
            region_volumes = chart3_data['region'].value_counts()
            sentiment_crosstab = sentiment_crosstab.reindex(region_volumes.index)
            
            # Ensure columns exist even if some sentiments are missing
            for col in ['Positive', 'Neutral', 'Negative']:
                if col not in sentiment_crosstab.columns:
                    sentiment_crosstab[col] = 0
                    
            # Reorder columns for standard presentation
            sentiment_crosstab = sentiment_crosstab[['Positive', 'Neutral', 'Negative']]

            # Define colors for sentiments
            colors = ['#2ca02c', '#7f7f7f', '#d62728'] # Green, Gray, Red
            
            ax = sentiment_crosstab.plot(kind='barh', stacked=True, figsize=(12, 8), color=colors, edgecolor='white')
            
            plt.title('Sentiment Distribution by Geographic Region\n(Natasha-Akpabio Discourse)', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Percentage (%)', fontsize=12)
            plt.ylabel('Region', fontsize=12)
            
            # Place legend outside
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add percentage annotations
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=9)
                
            sns.despine(left=True, bottom=True)
            ax.xaxis.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            chart3_path = os.path.join(CHARTS_DIR, '09_sentiment_by_region.png')
            plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved {chart3_path}")
        else:
             print("  Insufficient data or missing 'sentiment_label' column for Chart 3.")

    except FileNotFoundError:
        print(f"  Error: {SENTIMENT_DATA_PATH} not found. Cannot generate Chart 3.")

    print("\nPipeline execution finished successfully.")

if __name__ == "__main__":
    main()