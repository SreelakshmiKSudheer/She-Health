import pandas as pd

df = pd.read_csv(r'c:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\data\set_1\structured_endometriosis_data.csv')
print('Total rows:', len(df))
print('Columns:', df.columns.tolist())
print('Diagnosis counts:', df[' Diagnosis'].value_counts())
