import pandas as pd

th = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\set_1\\cleaned_dataset_Thyroid1.csv')
cer = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\set_1\\cervical-cancer_csv.csv')
pc = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\set_1\\PCOS_data.csv')
en = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\set_1\\structured_endometriosis_data.csv')

data = {
    'Thyroid': th.columns.tolist(),
    'Cervical Cancer': cer.columns.tolist(),
    'PCOS': pc.columns.tolist(),
    'Endometriosis': en.columns.tolist()
}

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
df.to_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\model_selection_and_validation\\featurePrint.csv', index=False)