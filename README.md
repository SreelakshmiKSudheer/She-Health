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
uvicorn app.main:app --host 172.16.9.41 --port 8000
```
Run the last two commands if the 3rd last one still shows error.

## Flutter Web Login Persistence
For web, local login data is stored per browser origin (host + port).
Use a fixed port when running Flutter web so previously registered users remain available:
```
cd shehealth
flutter run -d chrome --web-hostname 127.0.0.1 --web-port 8011
```