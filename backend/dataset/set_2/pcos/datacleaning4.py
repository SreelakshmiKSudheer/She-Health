import pandas as pd
import numpy as np

pcos = pd.read_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\set_2\pcos\data_cleaned.csv')
pcos.columns = pcos.columns.str.strip()

print(pcos.shape)
print(pcos['PCOS'].value_counts())
print(pcos.isnull().sum())
print(pcos.dtypes)

for col in [
    "Cycle Length",
    "Age",
    "City",
    "Overweight",
    "irregular or missed periods",
    "Difficulty in conceiving"
]:
    print(f"\n{col} unique values:")
    print(pcos[col].unique())

from sklearn.impute import SimpleImputer

df = pcos.copy()

# ---------------------------
# Strip whitespace everywhere
# ---------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

# ---------------------------
# Replace empty strings with NaN
# ---------------------------
df.replace("", np.nan, inplace=True)

# ---------------------------
# Convert numeric columns
# ---------------------------
numeric_cols = ["Cycle Length", "Age"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# Convert binary columns
# ---------------------------
binary_cols = [
    "Overweight",
    "irregular or missed periods",
    "Difficulty in conceiving"
]

for col in binary_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# City normalization
# ---------------------------
df["City"] = df["City"].str.lower()
df["City"] = df["City"].replace({
    "mumba": "mumbai"
})

# ---------------------------
# Verify result
# ---------------------------
print(df.dtypes)
print("\nMissing values after cleaning:\n", df.isnull().sum())
df.to_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\final_dataset\pcos.csv', index=False)

