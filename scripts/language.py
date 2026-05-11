import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from langdetect import detect, LangDetectException
 
 
# ── Language to Region Mapping ───────────────────────────────
# Maps detected language codes to broad geographic regions
# This mapping is used to group languages into world regions
LANGUAGE_REGION_MAP = {
    'en':  'English-Speaking World (USA, UK, Canada, Australia, Nigeria etc.)',
    'fr':  'Francophone Region (France, West Africa, Belgium, Canada)',
    'ar':  'Arab World (Middle East, North Africa)',
    'pt':  'Lusophone Region (Brazil, Portugal, Mozambique)',
    'es':  'Hispanic World (Spain, Latin America)',
    'de':  'German-Speaking Region (Germany, Austria, Switzerland)',
    'yo':  'Nigeria / West Africa (Yoruba)',
    'ha':  'Nigeria / West Africa (Hausa)',
    'ig':  'Nigeria / West Africa (Igbo)',
    'sw':  'East Africa (Swahili)',
    'af':  'Southern Africa (Afrikaans)',
    'zh-cn': 'China',
    'zh-tw': 'Taiwan / Chinese Diaspora',
    'ja':  'Japan',
    'ko':  'Korea',
    'hi':  'India / South Asia (Hindi)',
    'ru':  'Russia / Eastern Europe',
    'tr':  'Turkey',
    'nl':  'Netherlands / Belgium',
    'it':  'Italy',
    'pl':  'Poland',
    'id':  'Indonesia / Southeast Asia',
    'tl':  'Philippines',
    'ms':  'Malaysia / Singapore',
    'vi':  'Vietnam',
    'th':  'Thailand',
    'uk':  'Ukraine',
    'ro':  'Romania',
    'hu':  'Hungary',
    'cs':  'Czech Republic',
    'sk':  'Slovakia',
    'bg':  'Bulgaria',
    'hr':  'Croatia',
    'sr':  'Serbia',
    'el':  'Greece',
    'fi':  'Finland',
    'sv':  'Sweden / Scandinavia',
    'no':  'Norway',
    'da':  'Denmark',
    'he':  'Israel',
    'fa':  'Iran / Persian-Speaking Region',
    'ur':  'Pakistan / South Asia (Urdu)',
    'bn':  'Bangladesh / India (Bengali)',
    'ta':  'Tamil Nadu, India / Sri Lanka',
    'ml':  'Kerala, India (Malayalam)',
    'te':  'Andhra Pradesh, India (Telugu)',
    'am':  'Ethiopia (Amharic)',
    'so':  'Somalia / East Africa',
    'zu':  'South Africa (Zulu)',
    'xh':  'South Africa (Xhosa)',
}
 
 
def detect_language(text):
    try:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 'unknown'
        return detect(text)
    except LangDetectException:
        return 'unknown'
 
 
def map_to_region(lang_code):
    return LANGUAGE_REGION_MAP.get(lang_code, f'Other / Unknown ({lang_code})')
 
 
def main():
    # Load cleaned data
    df = pd.read_csv('data/cleaned/comments_cleaned.csv')
    print(f'Running language detection on {len(df)} comments...')
    print('This may take 30-60 seconds...')
 
    # Detect language for each comment
    df['language_code'] = df['text_clean'].apply(detect_language)
 
    # Map language codes to readable region names
    df['region_estimate'] = df['language_code'].apply(map_to_region)
 
    # ── Print Summary ─────────────────────────────────────────
    print('\n── Language Distribution ─────────────────────────────')
    lang_counts = df['language_code'].value_counts()
    print(lang_counts.head(15))
 
    print('\n── Regional Distribution ─────────────────────────────')
    region_counts = df['region_estimate'].value_counts()
    print(region_counts.head(15))
 
    print('\n── Regional Distribution by Outlet ───────────────────')
    print(df.groupby(['outlet', 'language_code']).size().unstack(fill_value=0))
 
    # ── Save Results ──────────────────────────────────────────
    df.to_csv('data/cleaned/comments_with_language.csv', index=False)
    print('\nSaved: data/cleaned/comments_with_language.csv')
 
    # ── Chart 1: Language Distribution Bar Chart ──────────────
    os.makedirs('outputs/charts', exist_ok=True)
    top_langs = lang_counts.head(10)
 
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(top_langs)),
        top_langs.values,
        color='#2E86AB',
        edgecolor='white'
    )
    ax.set_xticks(range(len(top_langs)))
    ax.set_xticklabels(
        [LANGUAGE_REGION_MAP.get(l, l) for l in top_langs.index],
        rotation=35,
        ha='right',
        fontsize=9
    )
    ax.set_title(
        'Language Distribution of Comments\nNatasha-Akpabio Saga — International Audience',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylabel('Number of Comments', fontsize=11)
    ax.set_xlabel('Detected Language / Region', fontsize=11)
 
    # Add count labels on top of each bar
    for bar, count in zip(bars, top_langs.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(count),
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
 
    plt.tight_layout()
    plt.savefig('outputs/charts/05_language_distribution.png', dpi=150)
    plt.close()
    print('Chart saved: outputs/charts/05_language_distribution.png')
 
    # ── Chart 2: Language vs Sentiment Heatmap ────────────────
    # Load sentiment data and merge
    sentiment_df = pd.read_csv('data/cleaned/comments_sentiment.csv')
    merged = sentiment_df.copy()
    merged['language_code'] = df['language_code']
    merged['region_estimate'] = df['region_estimate']
 
    # Filter to top 8 languages for readability
    top_8_langs = lang_counts.head(8).index.tolist()
    filtered = merged[merged['language_code'].isin(top_8_langs)]
 
    pivot = filtered.groupby(
        ['language_code', 'sentiment_label']
    ).size().unstack(fill_value=0)
 
    # Normalize to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Percentage of Comments (%)'}
    )
    ax.set_title(
        'Sentiment by Language Group (% of comments per language)\nNatasha-Akpabio Saga',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Sentiment Label', fontsize=11)
    ax.set_ylabel('Language Code', fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/charts/06_language_sentiment_heatmap.png', dpi=150)
    plt.close()
    print('Chart saved: outputs/charts/06_language_sentiment_heatmap.png')
 
    print('\nLanguage detection complete.')
 
 
if __name__ == '__main__':
    main()