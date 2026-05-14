import os 
import json 
import pandas as pd 
from flask import Flask, render_template, jsonify, request 
  
app = Flask(__name__) 
  
# Load sentiment data once when the server starts 
df = pd.read_csv('data/cleaned/comments_sentiment.csv') 
  
  
@app.route('/') 
def index(): 
    return render_template('index.html') 
  
  
@app.route('/api/summary')
def summary():
    total = len(df)
    counts = df['sentiment_label'].value_counts().to_dict()
    avg_compound = round(df['score_compound'].mean(), 4)

    by_outlet = df.groupby(
        ['outlet', 'sentiment_label']
    ).size().unstack(fill_value=0)
    
    # Convert to percentages for cleaner display
    outlet_data = by_outlet.to_dict(orient='index')

    return jsonify({
        'total_comments':   total,
        'sentiment_counts': counts,
        'average_compound': avg_compound,
        'by_outlet':        outlet_data
    })
  
  
@app.route('/api/comments') 
def comments(): 
    sentiment = request.args.get('sentiment', 'all') 
    outlet = request.args.get('outlet', 'all') 
    page = int(request.args.get('page', 1)) 
    per_page = 20 
  
    filtered = df.copy() 
  
    if sentiment != 'all': 
        filtered = filtered[filtered['sentiment_label'] == sentiment] 
  
    if outlet != 'all': 
        filtered = filtered[filtered['outlet'] == outlet] 
  
    total = len(filtered) 
    start = (page - 1) * per_page 
    end = start + per_page 
    page_data = filtered.iloc[start:end][[ 
        'outlet', 'text', 'sentiment_label', 'score_compound', 'likes' 
    ]].to_dict(orient='records') 
  
    return jsonify({ 
        'total': total, 
        'page': page, 
        'comments': page_data 
    }) 

@app.route('/api/language')
def language_data():
    lang_file = 'data/cleaned/comments_with_language.csv'

    if not os.path.exists(lang_file):
        return jsonify({
            'available': False,
            'message': 'Language data not yet generated. Run 05_language.py.'
        })

    df_lang = pd.read_csv(lang_file)

    if 'language_code' not in df_lang.columns:
        return jsonify({
            'available': False,
            'message': 'language_code column missing from file.'
        })

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
    return jsonify({
        'available': True,
        'labels': [LANGUAGE_LABELS.get(l, l) for l in lang_counts.index.tolist()],
        'values':  lang_counts.values.tolist()
    })


@app.route('/api/geo')
def geo_data():
    geo_file = 'data/cleaned/comments_with_geo.csv'

    if not os.path.exists(geo_file):
        return jsonify({
            'available': False,
            'message': 'Geographic data not yet generated.'
        })

    df_geo = pd.read_csv(geo_file)
    cols   = df_geo.columns.tolist()

    has_region = 'region' in cols
    has_name   = 'country_name' in cols

    if not has_region:
        return jsonify({
            'available': False,
            'message': 'Region column missing.'
        })

    total = len(df_geo)

    # Count verified as anything not Unknown
    if has_region:
        verified = int((
            df_geo['region'].notna() &
            (df_geo['region'] != 'Unknown')
        ).sum())
    else:
        verified = 0

    # Regional distribution — this is your real data
    region_data = {'labels': [], 'values': []}
    if has_region:
        rd = df_geo[
            df_geo['region'].notna() &
            (df_geo['region'] != 'Unknown')
        ]['region'].value_counts()
        region_data = {
            'labels': rd.index.tolist(),
            'values': rd.values.tolist()
        }

    # Language proxy distribution — relabel Proxy: XX to readable names
    PROXY_LABELS = {
        'Proxy: EN': 'English-Speaking',
        'Proxy: FR': 'French-Speaking',
        'Proxy: AR': 'Arabic-Speaking',
        'Proxy: PT': 'Portuguese-Speaking',
        'Proxy: ES': 'Spanish-Speaking',
        'Proxy: DE': 'German-Speaking',
        'Proxy: YO': 'Yoruba-Speaking',
        'Proxy: HA': 'Hausa-Speaking',
        'Proxy: IG': 'Igbo-Speaking',
        'Proxy: SW': 'Swahili-Speaking',
        'Proxy: SO': 'Somali-Speaking',
        'Proxy: AF': 'Afrikaans-Speaking',
        'Proxy: CY': 'Welsh-Speaking',
        'Proxy: ID': 'Indonesian-Speaking',
        'Proxy: TL': 'Filipino-Speaking',
        'Proxy: NL': 'Dutch-Speaking',
        'Proxy: NO': 'Norwegian-Speaking',
        'Proxy: RU': 'Russian-Speaking',
        'Proxy: ZH': 'Chinese-Speaking',
        'Proxy: HI': 'Hindi-Speaking',
    }

    country_data = {'labels': [], 'values': []}
    if has_name:
        cd = df_geo[
            df_geo['country_name'].notna()
        ]['country_name'].value_counts().head(10)

        readable_labels = [
            PROXY_LABELS.get(l, l) for l in cd.index.tolist()
        ]
        country_data = {
            'labels': readable_labels,
            'values': cd.values.tolist()
        }

    return jsonify({
        'available':    True,
        'total':        total,
        'verified':     verified,
        'coverage_pct': round(verified / total * 100, 1) if total > 0 else 0,
        'regions':      region_data,
        'countries':    country_data
    })
  
if __name__ == '__main__':
 # Read port from environment (Render sets this automatically)
 # Fall back to 5000 for local development
 port = int(os.environ.get('PORT', 5000))
 app.run(debug=False, host='0.0.0.0', port=port)