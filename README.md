# 🩺 She-Health

### *Empowering Preventive Women’s Healthcare Through ML-Based Risk Prediction*

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

* Predict risks of PCOD/PCOS, Endometriosis, and Cervical Cancer
* Provide personalised lifestyle and dietary recommendations
* Enable early awareness and preventive healthcare
* Generate health reports for medical consultations
* Ensure user-friendly access with strong data privacy and security

---

## Features

* Secure user access and data privacy
* Health data logging
* ML-Powered risk prediction
* Personalised guidance and reminders
* LLM-Based health report generation
* AI health chatbot

---

## 🧠 Machine Learning Models Used

The project compares multiple ML algorithms to identify the best-performing models.

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* AdaBoost

---

## 📊 Final Selected Models

### Endometriosis Prediction

#### ✅ Calibrated AdaBoost

**Performance Analysis**
* Accuracy: 0.880
* Precision: 0.867
* Recall: 0.915
* ROC-AUC: 0.962
* PR-AUC: 0.967
* Brier Score: 0.082

### PCOS Prediction

#### ✅ Calibrated XGBoost

**Performance Analysis**
* Accuracy: 0.796
* Precision: 0.919
* Recall: 0.810
* ROC-AUC: 0.875
* PR-AUC: 0.7521
* Brier Score: 0.1076

### Cervical Cancer Prediction

#### ✅ Calibrated Logistic Regression

**Performance Analysis**

* Accuracy: 0.860
* Precision: 0.956
* Recall: 0.893
* ROC-AUC: 0.672
* PR-AUC: 0.268
* Brier Score: 0.056

---

### Thyroid Disorder Prediction

#### ✅ AdaBoost

**Performance Analysis**

* Accuracy: 0.996
* Precision: 0.998
* Recall: 0.998
* ROC-AUC: 0.985
* PR-AUC: 0.964
* Brier Score: 0.0047

---

## ⚙️ Key Techniques Implemented

### ✅ Probability Calibration

Implemented using:

```python
CalibratedClassifierCV
```

Ensures predicted probabilities reflect real-world likelihoods.

---

### ✅ Threshold Optimization

Instead of using the default threshold (0.5), thresholds were tuned for:

* Higher Recall
* Better Healthcare Safety
* Reduced False Negatives

---

## 🛠️ Tech Stack

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
| MongoDB      | Server Storage        |
| FirebaseDB   | Chatbot Storage       |

---

## 📂 Project Structure

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

## 🔄 System Workflow

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

## 📈 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* F1-Score
* ROC-AUC
* PR-AUC
* Brier Score

---

## 🧪 Dataset Preprocessing

The datasets underwent extensive preprocessing including:

* Missing value handling
* Mean imputation
* Unknown class encoding
* Feature scaling using `StandardScaler`
* Feature selection
* Data cleaning and normalization

---

## 💻 Running the Project Locally

### Backend Setup

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
echo "*" > .venv\.gitignore
pip install "fastapi[standard]"
pip install -r requirements.txt
pip install uvicorn                             
```
#### Run the Backend
```bash
uvicorn app.main:app --reload
```
OR
```bash
uvicorn app.main:app --host <ip_address> --port 8000
```
---

### Frontend Setup

```bash
cd shehealth
flutter clean
flutter doctor
flutter pub get
flutter devices
flutter run
```

---

## 🔮 Future Enhancements

* Explainable AI (SHAP/LIME)
* Web & Mobile Deployment
* EHR Integration
* Real-time Clinical Support
* Personalized Health Recommendations
* Multi-disease Prediction System

---

## 🤝 Contributors

* [Sreelakshmi K](https://github.com/SreelakshmiKSudheer?utm_source=chatgpt.com)
* [Nanditha K M](https://github.com/NandithaRaveendranath?utm_source=chatgpt.com)
* [Delsa Davis](https://github.com/delsa-davis?utm_source=chatgpt.com)
* [MS Navaneetha](https://github.com/Navaneetha2504?utm_source=chatgpt.com)

---

## 🔗 Repository

[She-Health GitHub Repository](https://github.com/SreelakshmiKSudheer/She-Health?utm_source=chatgpt.com)

---

## 📌 Conclusion

She-Health demonstrates how Machine Learning can be effectively utilized for:

* Preventive healthcare
* Early disease detection
* Risk-based clinical decision support

By combining calibrated probability prediction, advanced ML algorithms, and clinically motivated optimization techniques, the project delivers a reliable and deployment-ready healthcare prediction system aimed at improving women’s healthcare accessibility and early screening support.
