import pandas as pd
import os

# Set pandas display options to show more columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Read the dataset
df = pd.read_excel('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\set_2\\endometriosis\\dataset.xlsx')

# Delete last column
df = df.drop(df.columns[[0, -2]], axis=1)

# Print the head
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# Save to cleaned1.csv in current directory
output_path = r'C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\set_2\\endometriosis\\cleaned1.csv'
df.to_csv(output_path, index=False, sep=',')
print(f"\nDataset saved to {output_path}")
