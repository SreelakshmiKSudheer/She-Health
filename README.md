# 🩺 She-Health

### *Empowering Preventive Women’s Healthcare Through AI-Driven Risk Prediction*

She-Health is an AI-powered healthcare prediction system designed to support **early detection and preventive screening** for critical women’s health conditions using Machine Learning.

The project focuses on:

* Cervical Cancer Risk Prediction
* Thyroid Disorder Prediction

By leveraging predictive analytics, calibrated probability estimation, and clinically meaningful risk categorization, She-Health aims to improve early diagnosis and healthcare accessibility.

---

## 🚀 Live Purpose of the Project

Many women’s health conditions remain undetected during their early stages due to:

* Lack of routine screening
* Delayed diagnosis
* Limited access to specialists
* Low awareness

She-Health addresses these challenges by developing an intelligent prediction system capable of:

* Identifying high-risk patients early
* Supporting preventive healthcare
* Assisting healthcare professionals in decision-making
* Reducing diagnostic delays

---

# ✨ Features

## 🧬 Cervical Cancer Risk Prediction

Predicts cervical cancer risk using:

* Medical history
* Reproductive health information
* Smoking habits
* STD history
* Screening test results

### Important Features

* Age
* Smoking history
* Number of sexual partners
* Hormonal contraceptive usage
* IUD usage
* Hinselmann test
* Schiller test
* Cytology results

---

## 🦋 Thyroid Disorder Prediction

Predicts thyroid abnormalities using hormonal and diagnostic measurements.

### Important Features

* TSH
* TT4
* FTI
* T4U
* T3
* Pregnancy information
* Thyroxine medication history
* Thyroid surgery history

---

# 🧠 Machine Learning Models Used

The project compares multiple ML algorithms to identify the best-performing models.

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* AdaBoost

---

# 📊 Final Selected Models

## Cervical Cancer Prediction

### ✅ Calibrated Logistic Regression

### Why?

* Clinically interpretable
* Reliable probability estimates
* Strong recall-oriented performance
* Better suited for imbalanced medical datasets

---

## Thyroid Disorder Prediction

### ✅ AdaBoost

### Performance Highlights

* Accuracy ≈ 99.6%
* ROC-AUC ≈ 0.986
* PR-AUC ≈ 0.964
* Very low Brier Score

---

# ⚙️ Key Techniques Implemented

## ✅ Probability Calibration

Implemented using:

```python
CalibratedClassifierCV
```

Ensures predicted probabilities reflect real-world likelihoods.

---

## ✅ Threshold Optimization

Instead of using the default threshold (0.5), thresholds were tuned for:

* Higher Recall
* Better Healthcare Safety
* Reduced False Negatives

---

## ✅ Risk Categorization

| Probability Range | Risk Level     |
| ----------------- | -------------- |
| < 5%              | No Risk        |
| 5–10%             | Low Risk       |
| 10–25%            | Moderate Risk  |
| 25–50%            | High Risk      |
| > 50%             | Very High Risk |

---

# 🛠️ Tech Stack

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Backend & ML          |
| Pandas       | Data Processing       |
| NumPy        | Numerical Computation |
| Scikit-learn | Machine Learning      |
| XGBoost      | Gradient Boosting     |
| LightGBM     | Boosting Framework    |
| Joblib       | Model Serialization   |
| Flutter      | Frontend Application  |
| Dart         | Frontend Development  |

---

# 📂 Project Structure

```bash
She-Health/
│
├── backend/
│   ├── app/
│   ├── services/
│   ├── routes/
│   ├── ml/
│   ├── dataset/
│   └── tests/
│
├── shehealth/
│   ├── lib/
│   ├── android/
│   ├── ios/
│   ├── web/
│   └── pubspec.yaml
│
└── README.md
```

---

# 🔄 System Workflow

```text
Patient Data
      ↓
Data Cleaning
      ↓
Preprocessing & Missing Value Handling
      ↓
Feature Selection
      ↓
Model Training
      ↓
Probability Calibration
      ↓
Threshold Optimization
      ↓
Risk Prediction
      ↓
Risk Categorization
      ↓
Clinical Decision Support
```

---

# 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1-Score
* ROC-AUC
* PR-AUC
* Brier Score

---

# 🧪 Dataset Preprocessing

The datasets underwent extensive preprocessing including:

* Missing value handling
* Mean imputation
* Unknown class encoding
* Feature scaling using `StandardScaler`
* Feature selection
* Data cleaning and normalization

---

# 💻 Running the Project Locally

## Backend Setup

```bash
cd backend

python -m venv .venv

# Activate environment
# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python app/main.py
```

---

## Frontend Setup

```bash
cd shehealth

flutter pub get

flutter run
```

---

# 🔮 Future Enhancements

* Explainable AI (SHAP/LIME)
* Web & Mobile Deployment
* EHR Integration
* Real-time Clinical Support
* Personalized Health Recommendations
* Multi-disease Prediction System

---

# 🤝 Contributors

* [Sreelakshmi K](https://github.com/SreelakshmiKSudheer?utm_source=chatgpt.com)
* [Nanditha K M](https://github.com/NandithaRaveendranath?utm_source=chatgpt.com)
* [Delsa Davis](https://github.com/delsa-davis?utm_source=chatgpt.com)
* [MS Navaneetha](https://github.com/Navaneetha2504?utm_source=chatgpt.com)

---

# 🔗 Repository

[She-Health GitHub Repository](https://github.com/SreelakshmiKSudheer/She-Health?utm_source=chatgpt.com)

---

# 📌 Conclusion

She-Health demonstrates how Machine Learning can be effectively utilized for:

* Preventive healthcare
* Early disease detection
* Risk-based clinical decision support

By combining calibrated probability prediction, advanced ML algorithms, and clinically motivated optimization techniques, the project delivers a reliable and deployment-ready healthcare prediction system aimed at improving women’s healthcare accessibility and early screening support.

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
uvicorn app.main:app --host 10.186.204.75 --port 8000
```
Run the last two commands if the 3rd last one still shows error.

## Flutter Web Login Persistence
For web, local login data is stored per browser origin (host + port).
Use a fixed port when running Flutter web so previously registered users remain available:
```
cd shehealth
flutter run -d chrome --web-hostname 127.0.0.1 --web-port 8011
flutter clean
flutter doctor
flutter pub get
flutter devices
flutter run
```