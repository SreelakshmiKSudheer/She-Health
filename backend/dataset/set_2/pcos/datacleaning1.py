import pandas as pd
import numpy as np
import os

data = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\navaneetha\\Major Project\\She-Health\\backend\\data\\set_2\\pcos\\results.csv')

data.columns = data.columns.str.strip()  # Remove trailing spaces from column names

data.drop('Timestamp', inplace=True, axis=1)
data.drop('PCOS tested', inplace=True, axis=1)
data.drop('When do you experience mood swings?', inplace=True, axis=1)

data["City"] = data["City"].str.lower()  # lower all city names

data = data.rename(columns={'PCOS from age of': 'PCOS_from'})

data['PCOS_from'] = data.PCOS_from.str.extract(r'(\d+)')

data.to_csv('C:\\Users\\Lenovo\\Desktop\\navaneetha\\Major Project\\She-Health\\backend\\data\\set_2\\pcos\\allData.csv', index=False)

PCOS_True = data[data['PCOS'].str.strip() == 'Yes']
PCOS_True = PCOS_True.dropna(subset=["PCOS_from"])

PCOS_True.to_csv('C:\\Users\\Lenovo\\Desktop\\navaneetha\\Major Project\\She-Health\\backend\\data\\set_2\\pcos\\OnlyPCOS.csv', index=False)
