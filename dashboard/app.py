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
        return jsonify({'error': 'Language data not available. Run 05_language.py first.'})

    df_lang = pd.read_csv(lang_file)

    LANGUAGE_LABELS = {
        'en': 'English', 'fr': 'French', 'ar': 'Arabic',
        'pt': 'Portuguese', 'es': 'Spanish', 'de': 'German',
        'yo': 'Yoruba', 'ha': 'Hausa', 'ig': 'Igbo',
        'sw': 'Swahili', 'hi': 'Hindi', 'zh-cn': 'Chinese',
        'ru': 'Russian', 'id': 'Indonesian', 'tl': 'Filipino',
    }

    lang_counts = df_lang['language_code'].value_counts().head(10)
    return jsonify({
        'labels': [LANGUAGE_LABELS.get(l, l) for l in lang_counts.index.tolist()],
        'values': lang_counts.values.tolist()
    })


@app.route('/api/geo')
def geo_data():
    geo_file = 'data/cleaned/comments_with_geo.csv'
    if not os.path.exists(geo_file):
        return jsonify({'error': 'Geographic data not available. Run 06_geo_lookup.py first.'})

    df_geo = pd.read_csv(geo_file)

    # Regional distribution
    region_data = df_geo[
        df_geo['region'].notna() & (df_geo['region'] != 'Unknown')
    ]['region'].value_counts()

    # Top countries
    country_data = df_geo[
        df_geo['country_name'].notna() &
        (df_geo['country_name'] != 'Not Specified')
    ]['country_name'].value_counts().head(10)

    total = len(df_geo)
    verified = int(df_geo['country_code'].notna().sum())

    return jsonify({
        'total': total,
        'verified': verified,
        'coverage_pct': round(verified / total * 100, 1),
        'regions': {
            'labels': region_data.index.tolist(),
            'values': region_data.values.tolist()
        },
        'countries': {
            'labels': country_data.index.tolist(),
            'values': country_data.values.tolist()
        }
    })  
  
if __name__ == '__main__': 
    app.run(debug=True, port=5000)