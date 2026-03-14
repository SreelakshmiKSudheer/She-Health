import pandas as pd
import os

# Get the base path relative to the current script
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Define dataset paths
cer = pd.read_csv(os.path.join(base_path, 'dataset', 'final_dataset', 'cervical_cancer.csv'))
pc = pd.read_csv(os.path.join(base_path, 'dataset', 'final_dataset', 'pcos.csv'))
en = pd.read_csv(os.path.join(base_path, 'dataset', 'final_dataset', 'endometriosis.csv'))

data = {
    'Cervical Cancer': cer.columns.tolist(),
    'PCOS': pc.columns.tolist(),
    'Endometriosis': en.columns.tolist()
}

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

# Define output path
output_path = os.path.join(os.path.dirname(__file__), 'featurePrint.csv')
df.to_csv(output_path, index=False)