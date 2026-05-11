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
    if not os.path.exists('data/cleaned/comments_with_language.csv'):
        print('Language data not found. Run 05_language.py first.')
        return

    df_lang = pd.read_csv('data/cleaned/comments_with_language.csv')
    lang_counts = df_lang['language_code'].value_counts().head(10)

    LANGUAGE_LABELS = {
        'en': 'English', 'fr': 'French', 'ar': 'Arabic',
        'pt': 'Portuguese', 'es': 'Spanish', 'de': 'German',
        'yo': 'Yoruba', 'ha': 'Hausa', 'ig': 'Igbo',
        'sw': 'Swahili', 'hi': 'Hindi', 'zh-cn': 'Chinese',
        'ru': 'Russian', 'id': 'Indonesian', 'tl': 'Filipino',
    }

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


# ── Chart 6: Regional Distribution ───────────────────────────
def chart_regional_distribution():
    if not os.path.exists('data/cleaned/comments_with_geo.csv'):
        print('Geographic data not found. Run 06_geo_lookup.py first.')
        return

    df_geo = pd.read_csv('data/cleaned/comments_with_geo.csv')
    region_data = df_geo[
        df_geo['region'].notna() & (df_geo['region'] != 'Unknown')
    ]['region'].value_counts()

    if len(region_data) == 0:
        print('No verified geographic data found. Skipping regional chart.')
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
    plt.savefig('outputs/charts/06_regional_distribution.png', dpi=150)
    plt.close()
    print('Chart 6 saved: Regional Distribution')


# ── Chart 7: Top Countries ────────────────────────────────────
def chart_top_countries():
    if not os.path.exists('data/cleaned/comments_with_geo.csv'):
        print('Geographic data not found. Run 06_geo_lookup.py first.')
        return

    df_geo = pd.read_csv('data/cleaned/comments_with_geo.csv')
    country_counts = df_geo[
        df_geo['country_name'].notna() &
        (df_geo['country_name'] != 'Not Specified')
    ]['country_name'].value_counts().head(15)

    if len(country_counts) == 0:
        print('No verified country data found. Skipping country chart.')
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
    plt.savefig('outputs/charts/07_top_countries.png', dpi=150)
    plt.close()
    print('Chart 7 saved: Top Countries')
  
# Run all charts
chart_overall_distribution()
chart_outlet_comparison()
chart_compound_histogram()
chart_wordcloud()
chart_language_distribution()
chart_regional_distribution()
chart_top_countries()

print('\nAll charts generated in outputs/charts/')