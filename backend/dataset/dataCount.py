import pandas as pd

# Load datasets (use raw strings to avoid escape issues)
cervical = pd.read_csv(r'C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\dataset\\final_dataset\\cervical_cancer.csv')
pcos = pd.read_csv(r'C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\dataset\\final_dataset\\pcos.csv')
# Use structured endometriosis dataset that contains binary labels
endometriosis = pd.read_csv(r'C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\dataset\\final_dataset\\endometriosis.csv')

# Strip column names to remove leading/trailing spaces
cervical.columns = cervical.columns.str.strip()
pcos.columns = pcos.columns.str.strip()
endometriosis.columns = endometriosis.columns.str.strip()

print("Cervical Cancer Dataset:")
print("Total rows:", len(cervical))
if 'Biopsy' in cervical.columns:
    print(cervical['Biopsy'].value_counts())
else:
    print("Biopsy column not found. Available columns:", cervical.columns.tolist())

print("\nPCOS Dataset:")
print("Total rows:", len(pcos))
if 'PCOS' in pcos.columns:
    print(pcos['PCOS'].value_counts())
else:
    print("PCOS column not found. Available columns:", pcos.columns.tolist())

print("\nEndometriosis Dataset:")
print("Total rows:", len(endometriosis))
target_col = 'Diagnosis'
if target_col in endometriosis.columns:
    print(endometriosis[target_col].value_counts())
else:
    print(f"{target_col} column not found. Available columns:", endometriosis.columns.tolist())