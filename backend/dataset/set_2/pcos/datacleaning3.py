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

for col in binary:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

df['Difficulty in conceiving'] = df['Difficulty in conceiving'].map({'Not Applicable': 0, 'Yes': 1})

for col in ternary:
    df[col] = df[col].map({'normal': 0, 'moderate': 1, 'excessive': 2})

print(df.dtypes)

df.to_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\set_2\pcos\data_cleaned.csv', index=False)