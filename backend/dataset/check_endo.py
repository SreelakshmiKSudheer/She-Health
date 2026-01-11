import pandas as pd

df = pd.read_csv('c:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\data\final_dataset\endometriosis.csv')
print("Total rows:", len(df))
print("\nColumns with non-null values:")
for col in df.columns:
    non_null = df[col].notna().sum()
    if non_null > 0:
        print(f"{col}: {non_null} non-null, unique: {df[col].nunique()}")
