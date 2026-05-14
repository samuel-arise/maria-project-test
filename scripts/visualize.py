import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches 
import seaborn as sns 
from wordcloud import WordCloud 
  
# Load sentiment results 
df = pd.read_csv('data/cleaned/comments_sentiment.csv') 
os.makedirs('outputs/charts', exist_ok=True) 
  
# Colour scheme 
COLORS = { 
    'Positive': '#27AE60', 
    'Neutral':  '#F39C12', 
    'Negative': '#E74C3C' 
} 
  
  
# ── Chart 1: Overall Sentiment Distribution ────────────────── 
def chart_overall_distribution(): 
    counts = df['sentiment_label'].value_counts() 
    colors = [COLORS[label] for label in counts.index] 
  
    fig, ax = plt.subplots(figsize=(8, 8)) 
    wedges, texts, autotexts = ax.pie( 
        counts, 
        labels=counts.index, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=140, 
        textprops={'fontsize': 13} 
    ) 
    ax.set_title( 
        'Overall Sentiment Distribution\nNatasha-Akpabio Saga (International Audience)', 
        fontsize=14, fontweight='bold', pad=20 
    ) 
    plt.tight_layout() 
    plt.savefig('outputs/charts/01_overall_distribution.png', dpi=150) 
    plt.close() 
    print('Chart 1 saved: Overall Distribution') 
  
  
# ── Chart 2: BBC vs Al Jazeera Comparison ──────────────────── 
def chart_outlet_comparison(): 
    grouped = df.groupby(['outlet', 'sentiment_label']).size().unstack(fill_value=0) 
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100 
  
    ax = grouped_pct.plot( 
        kind='bar', 
        color=[COLORS.get(c, '#888888') for c in grouped_pct.columns], 
        figsize=(10, 6), 
        edgecolor='white', 
        width=0.6 
    ) 
    ax.set_title( 
        'Sentiment Comparison: BBC vs Al Jazeera\n(% of comments per outlet)', 
        fontsize=14, fontweight='bold' 
    ) 
    ax.set_xlabel('Media Outlet', fontsize=12) 
    ax.set_ylabel('Percentage of Comments (%)', fontsize=12) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 
    ax.legend(title='Sentiment') 
    plt.tight_layout() 
    plt.savefig('outputs/charts/02_outlet_comparison.png', dpi=150) 
    plt.close() 
    print('Chart 2 saved: Outlet Comparison') 
  
  
# ── Chart 3: Compound Score Histogram ──────────────────────── 
def chart_compound_histogram(): 
    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.hist(df['score_compound'], bins=40, color='#2E86AB', edgecolor='white') 
    ax.axvline(x=0.05,  color='#27AE60', linestyle='--', label='Positive threshold') 
    ax.axvline(x=-0.05, color='#E74C3C', linestyle='--', label='Negative threshold') 
    ax.set_title('Distribution of Compound Sentiment Scores', fontsize=14, fontweight='bold') 
    ax.set_xlabel('Compound Score (-1 = Most Negative, +1 = Most Positive)', fontsize=11) 
    ax.set_ylabel('Number of Comments', fontsize=11) 
    ax.legend() 
    plt.tight_layout() 
    plt.savefig('outputs/charts/03_compound_histogram.png', dpi=150) 
    plt.close() 
    print('Chart 3 saved: Compound Score Histogram') 
  
  
# ── Chart 4: Word Cloud ─────────────────────────────────────── 
def chart_wordcloud(): 
    text = ' '.join(df['text_clean'].dropna().tolist()) 
    wordcloud = WordCloud( 
        width=1200, height=600, 
        background_color='white', 
        colormap='RdYlGn', 
        max_words=150 
    ).generate(text) 
  
    fig, ax = plt.subplots(figsize=(14, 7)) 
    ax.imshow(wordcloud, interpolation='bilinear') 
    ax.axis('off') 
    ax.set_title( 
        'Most Frequent Words in International Audience Comments', 
        fontsize=14, fontweight='bold' 
    ) 
    plt.tight_layout() 
    plt.savefig('outputs/charts/04_wordcloud.png', dpi=150) 
    plt.close() 
    print('Chart 4 saved: Word Cloud') 

# ── Chart 5: Language Distribution ───────────────────────────
def chart_language_distribution():
    lang_file = 'data/cleaned/comments_with_language.csv'

    if not os.path.exists(lang_file):
        print('Chart 5 skipped: comments_with_language.csv not found.')
        return

    df_lang = pd.read_csv(lang_file)

    if 'language_code' not in df_lang.columns:
        print('Chart 5 skipped: language_code column not found.')
        return

    LANGUAGE_LABELS = {
    'en': 'English',
    'fr': 'French',
    'ar': 'Arabic',
    'pt': 'Portuguese',
    'es': 'Spanish',
    'de': 'German',
    'yo': 'Yoruba',
    'ha': 'Hausa',
    'ig': 'Igbo',
    'sw': 'Swahili',
    'hi': 'Hindi',
    'zh-cn': 'Chinese',
    'ru': 'Russian',
    'id': 'Indonesian',
    'tl': 'Filipino',
    'af': 'Afrikaans',
    'so': 'Somali',
    'cy': 'Welsh',
    'da': 'Danish',
    'nl': 'Dutch',
    'it': 'Italian',
    'tr': 'Turkish',
    'ko': 'Korean',
    'ja': 'Japanese',
    'vi': 'Vietnamese',
    'pl': 'Polish',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'ro': 'Romanian',
    'uk': 'Ukrainian',
    'ms': 'Malay',
    'th': 'Thai',
    'ur': 'Urdu',
    'bn': 'Bengali',
    'am': 'Amharic',
    'zu': 'Zulu',
    'xh': 'Xhosa',
    'ca': 'Catalan',
    'hr': 'Croatian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'bg': 'Bulgarian',
    'sr': 'Serbian',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'et': 'Estonian',
    'he': 'Hebrew',
    'fa': 'Persian',
    'unknown': 'Unknown',
}

    lang_counts = df_lang['language_code'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(lang_counts)),
        lang_counts.values,
        color='#8E44AD',
        edgecolor='white'
    )
    ax.set_xticks(range(len(lang_counts)))
    ax.set_xticklabels(
        [LANGUAGE_LABELS.get(l, l) for l in lang_counts.index],
        rotation=30, ha='right', fontsize=10
    )
    ax.set_title(
        'Language Distribution of Comments\nNatasha-Akpabio Saga — International Audience',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylabel('Number of Comments', fontsize=11)
    ax.set_xlabel('Detected Language', fontsize=11)

    for bar, count in zip(bars, lang_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(count),
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig('outputs/charts/05_language_distribution.png', dpi=150)
    plt.close()
    print('Chart 5 saved: Language Distribution')


# ── Chart 6: Language vs Sentiment Heatmap ───────────────────
def chart_language_sentiment_heatmap():
    lang_file      = 'data/cleaned/comments_with_language.csv'
    sentiment_file = 'data/cleaned/comments_sentiment.csv'

    if not os.path.exists(lang_file):
        print('Chart 6 skipped: comments_with_language.csv not found.')
        return

    if not os.path.exists(sentiment_file):
        print('Chart 6 skipped: comments_sentiment.csv not found.')
        return

    df_lang      = pd.read_csv(lang_file)
    df_sentiment = pd.read_csv(sentiment_file)

    if 'language_code' not in df_lang.columns:
        print('Chart 6 skipped: language_code column missing.')
        return

    # Merge language into sentiment dataframe by index
    df_merged = df_sentiment.copy()
    df_merged['language_code'] = df_lang['language_code'].values

    # Keep only top 8 languages for readability
    top_langs = df_merged['language_code'].value_counts().head(8).index.tolist()
    df_filtered = df_merged[df_merged['language_code'].isin(top_langs)]

    pivot = df_filtered.groupby(
        ['language_code', 'sentiment_label']
    ).size().unstack(fill_value=0)

    # Normalize to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    import seaborn as sns
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
        'Sentiment by Language Group\n(% of comments per language)',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('Sentiment Label', fontsize=11)
    ax.set_ylabel('Language Code', fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/charts/06_language_sentiment_heatmap.png', dpi=150)
    plt.close()
    print('Chart 6 saved: Language Sentiment Heatmap')


# ── Chart 7: Regional Distribution Pie Chart ─────────────────
def chart_regional_distribution():
    geo_file = 'data/cleaned/comments_with_geo.csv'

    if not os.path.exists(geo_file):
        print('Chart 7 skipped: comments_with_geo.csv not found.')
        return

    df_geo = pd.read_csv(geo_file)

    if 'region' not in df_geo.columns:
        print('Chart 7 skipped: region column missing.')
        return

    region_data = df_geo[
        df_geo['region'].notna() & (df_geo['region'] != 'Unknown')
    ]['region'].value_counts()

    if len(region_data) == 0:
        print('Chart 7 skipped: no verified region data found.')
        return

    colors = [
        '#2E86AB','#27AE60','#E74C3C','#F39C12','#8E44AD',
        '#1ABC9C','#E67E22','#2C3E50','#D35400','#7F8C8D'
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(
        region_data.values,
        labels=region_data.index,
        colors=colors[:len(region_data)],
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 10}
    )
    ax.set_title(
        'Regional Distribution of Commenters\n(Verified YouTube Channel Country Data)',
        fontsize=14, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig('outputs/charts/07_regional_distribution.png', dpi=150)
    plt.close()
    print('Chart 7 saved: Regional Distribution')


# ── Chart 8: Top 15 Countries ─────────────────────────────────
def chart_top_countries():
    geo_file = 'data/cleaned/comments_with_geo.csv'

    if not os.path.exists(geo_file):
        print('Chart 8 skipped: comments_with_geo.csv not found.')
        return

    df_geo = pd.read_csv(geo_file)

    if 'country_name' not in df_geo.columns:
        print('Chart 8 skipped: country_name column missing.')
        return

    country_counts = df_geo[
        df_geo['country_name'].notna() &
        (df_geo['country_name'] != 'Not Specified')
    ]['country_name'].value_counts().head(15)

    if len(country_counts) == 0:
        print('Chart 8 skipped: no verified country data found.')
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        range(len(country_counts)),
        country_counts.values,
        color='#2E86AB',
        edgecolor='white'
    )
    ax.set_yticks(range(len(country_counts)))
    ax.set_yticklabels(country_counts.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(
        'Top 15 Countries — Verified Commenter Locations\nNatasha-Akpabio Saga Coverage',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Number of Commenters', fontsize=11)

    for bar, count in zip(bars, country_counts.values):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va='center', fontsize=9, fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig('outputs/charts/08_top_countries.png', dpi=150)
    plt.close()
    print('Chart 8 saved: Top Countries')


# ── Chart 9: Sentiment by Region ─────────────────────────────
def chart_sentiment_by_region():
    geo_file       = 'data/cleaned/comments_with_geo.csv'
    sentiment_file = 'data/cleaned/comments_sentiment.csv'

    if not os.path.exists(geo_file):
        print('Chart 9 skipped: comments_with_geo.csv not found.')
        return

    if not os.path.exists(sentiment_file):
        print('Chart 9 skipped: comments_sentiment.csv not found.')
        return

    df_geo       = pd.read_csv(geo_file)
    df_sentiment = pd.read_csv(sentiment_file)

    if 'region' not in df_geo.columns:
        print('Chart 9 skipped: region column missing.')
        return

    # Merge region into sentiment data by index
    df_merged = df_sentiment.copy()
    df_merged['region'] = df_geo['region'].values

    df_filtered = df_merged[
        df_merged['region'].notna() & (df_merged['region'] != 'Unknown')
    ]

    if len(df_filtered) == 0:
        print('Chart 9 skipped: no region data available to plot.')
        return

    pivot = df_filtered.groupby(
        ['region', 'sentiment_label']
    ).size().unstack(fill_value=0)

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    ax = pivot_pct.plot(
        kind='bar',
        color=['#E74C3C', '#F39C12', '#27AE60'],
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
    print('Chart 9 saved: Sentiment by Region')


# ── Run All Charts ────────────────────────────────────────────
chart_overall_distribution()
chart_outlet_comparison()
chart_compound_histogram()
chart_wordcloud()
chart_language_distribution()
chart_language_sentiment_heatmap()
chart_regional_distribution()
chart_top_countries()
chart_sentiment_by_region()

print('\nAll available charts generated in outputs/charts/')