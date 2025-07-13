import pandas as pd
df = pd.read_csv('data/processed/tier2_positional_accuracy.csv')
print('Columns:', list(df.columns))
print('Total rows:', len(df))
print('Unique resolved anchors:', df['anchor_feature_name'].nunique() if 'anchor_feature_name' in df else 'N/A')
good = df[~df.get('notes', pd.Series()).astype(str).str.contains('bad match|discard', case=False, na=False) & ~df.get('manual_lat_lon', pd.Series()).astype(str).str.contains('bad match|discard', case=False, na=False) & df['final_d_km'].notna()]
print('Good anchors count:', len(good))
if good.empty:
    print('No good anchors')
else:
    print(good['final_d_km'].describe())
    print('P90:', good['final_d_km'].quantile(0.9)) 