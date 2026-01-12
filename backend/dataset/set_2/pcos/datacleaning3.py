import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\set_2\pcos\data_final.csv')
df.columns = df.columns.str.strip()

binary = ['PCOS', 
         'Overweight', 
         'loss weight gain / weight loss', 
         'irregular or missed periods', 
         'Acne or skin tags', 
         'Hair thinning or hair loss', 
         'Dark patches', 
         'always tired', 
         'more Mood Swings', 
         'canned food often']

ternary = ['Hair growth  on Cheeks', 
           'Hair growth Between breasts',
           'Hair growth  on Upper lips',
           'Hair growth in Arms',
           'Hair growth on Inner thighs']

# show sample unique values to help debug mismatches
for col in binary + ['Difficulty in conceiving'] + ternary:
    if col in df.columns:
        print(col, pd.Series(df[col].dropna().unique()).head(10).tolist())

def map_values(df, col, mapping):
    if col not in df.columns:
        print(f"Warning: column '{col}' not found in dataframe")
        return None
    lowered_map = {str(k).strip().lower(): v for k, v in mapping.items()}
    s = df[col].astype(str).str.strip().str.lower()
    return s.map(lowered_map)

for col in binary:
    res = map_values(df, col, {'No': 0, 'Yes': 1})
    if res is not None:
        df[col] = res.astype('Int64')

res = map_values(df, 'Difficulty in conceiving', {'Not Applicable': 0, 'Yes': 1})
if res is not None:
    df['Difficulty in conceiving'] = res.astype('Int64')

for col in ternary:
    res = map_values(df, col, {'normal': 0, 'moderate': 1, 'excessive': 2})
    if res is not None:
        df[col] = res.astype('Int64')

print(df.dtypes)
print(df.head())

df.to_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\set_2\pcos\data_cleaned.csv', index=False)
df.to_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\final_dataset\pcos.csv', index=False)