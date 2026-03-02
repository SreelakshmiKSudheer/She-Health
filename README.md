# She-Health
## Backend Setup and Execution
```
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
echo "*" > .venv\.gitignore
pip install "fastapi[standard]"
pip install -r requirements.txt
fastapi dev app.main.py
pip install uvicorn                             
uvicorn app.main:app --reload
```
Run the last two commands if the 3rd last one still shows error.