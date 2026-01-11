import numpy as np
import pandas as pd
import os

# Path to allData.csv in the workspace root
path = "C:\\Users\\Lenovo\\Desktop\\navaneetha\\Major Project\\She-Health\\allData.csv"
data = pd.read_csv(path)

# Strip spaces from column names and string values
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.strip()

# Function to label 'Hair growth on Chin'
def label5(row):
    if row['Hair growth on Chin'] == 'normal':
        return 0
    elif row['Hair growth on Chin'] == 'moderate':
        return 1
    else:
        return 2

data['Hair growth on Chin'] = data.apply(lambda row: label5(row), axis=1)

# Function to label 'relocated city'
def label16(row):
    if row['relocated city'] == 'Yes':
        return 1
    else:
        return 0

data['relocated city'] = data.apply(lambda row: label16(row), axis=1)

# Function to label 'Period Length'
def label17(row):
    if row['Period Length'] == '2-3 days':
        return 3
    elif row['Period Length'] == '4-5 days':
        return 5
    elif row['Period Length'] == '6-7 days':
        return 7
    else:
        return 9

data['Period Length'] = data.apply(lambda row: label17(row), axis=1)

# Function to label 'Cycle Length'
def label18(row):
    if row['Cycle Length'] == '20-24 days':
        return 22
    elif row['Cycle Length'] == '20-28 days':
        return 25
    elif row['Cycle Length'] == '25-28':
        return 27
    elif row['Cycle Length'] == '29-35 days':
        return 32
    elif row['Cycle Length'] == '36+ days':
        return 37
    else:
        return None  # Use None instead of 'NaN'

data['Cycle Length'] = data.apply(lambda row: label18(row), axis=1)

# Remove PCOS_from column if it exists
if 'PCOS_from' in data.columns:
    del data['PCOS_from']

# Convert `Age` to cleaned strings and map to nullable integers
age_map = {
    'Below 18': 1,
    '18-25': 2,
    '26-30': 3,
    '31-35': 4,
    '36-40': 5,
    '41-45': 6,
    'Above 45': 7
}

if 'Age' in data.columns:
    # ensure strings, strip whitespace, then map
    data['Age'] = data['Age'].astype(str).str.strip()
    data['Age'] = data['Age'].map(age_map)
    # use pandas nullable integer dtype to allow missing values
    data['Age'] = data['Age'].astype('Int64')

# `Age` has been converted to nullable integers above; no further mapping required.

# Save the cleaned data in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
data.to_csv(os.path.join(script_dir, 'data_final.csv'), index=False)

print("Data cleaned and saved to data_final.csv")
