import pandas as pd
import os

def main():
    # Load YouTube data
    yt = pd.read_csv('data/raw/comments_raw.csv')
    yt['source'] = 'YouTube'

    # Load converted X data
    x = pd.read_csv('data/raw/x_apify_cleaned.csv')
    x['source'] = 'x_data'

    # Merge
    merged = pd.concat([yt, x], ignore_index=True)
    merged = merged.drop_duplicates(subset='text')

    print(f'YouTube comments:  {len(yt)}')
    print(f'X tweets:          {len(x)}')
    print(f'Merged total:      {len(merged)}')
    print(f'After dedup:       {len(merged)}')

    merged.to_csv('data/raw/comments_raw.csv', index=False)
    print('\nMerged dataset saved to data/raw/comments_raw.csv')
    print('Run the rest of the pipeline as normal from clean.py')


if __name__ == '__main__':
    main()