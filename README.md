🏦 Loan Default Risk Prediction System

A production-ready machine learning web application that predicts whether a loan applicant is likely to default. The project uses a Stacking Ensemble Machine Learning Model, a FastAPI backend, and a Streamlit frontend, fully containerized using Docker and deployed to the cloud.

---

🚀 Live Demo

Link : https://frontend-pydr.onrender.com/

---

📌 Project Overview

Financial institutions receive thousands of loan applications every day. Approving risky applicants can lead to financial losses, while rejecting reliable applicants reduces business opportunities.

This application predicts the probability of loan default and allows users to adjust the decision threshold based on business requirements.

---

✨ Features

- 📊 Loan Default Prediction
- 🎯 Adjustable Prediction Threshold
- 📈 Dynamic Precision, Recall and F1 Score
- 📋 Loan Summary Display
- 🤖 Stacking Ensemble Machine Learning Model
- ⚡ FastAPI REST API
- 🎨 Interactive Streamlit Dashboard
- 🐳 Dockerized Deployment
- ☁️ Cloud Deployment Ready

---

🧠 Machine Learning Pipeline

The prediction pipeline consists of:

- Data Preprocessing
- Feature Encoding
- Feature Scaling
- Stacking Ensemble Model
- Probability Prediction
- Threshold-based Decision Making

---

🛠️ Tech Stack

Machine Learning

- Python
- Scikit-learn
- NumPy
- Pandas

Backend

- FastAPI
- Pydantic
- Uvicorn

Frontend

- Streamlit

Deployment

- Docker
- Docker Compose
- Render

---

📊 Input Features

- Annual Income
- Loan Amount
- Interest Rate
- Employment Length
- Debt-to-Income Ratio
- Loan Term
- Home Ownership
- Loan Purpose
- Total Credit Accounts

---

📈 Output

The system returns:

- Prediction (SAFE / UNSAFE)
- Risk Probability
- Selected Threshold
- Precision
- Recall
- F1 Score
- Threshold Metrics Table

---

⚙️ Local Setup

Build Docker images

docker compose build

Run containers

docker compose up

Frontend:

http://localhost:8501

Backend API:

http://localhost:8000

Swagger Documentation:

http://localhost:8000/docs

---

🔍 API Endpoint

POST "/predict"

Example Request

{
  "annual_inc": 50000,
  "loan_amnt": 10000,
  "int_rate": 10,
  "emp_length": 5,
  "dti": 0.5,
  "term": " 36 months",
  "home_ownership": "MORTGAGE",
  "purpose": "debt_consolidation",
  "total_acc": 5,
  "selected_threshold": 0.5
}

---

📦 Deployment

The application is fully containerized using Docker and deployed as separate frontend and backend services.

---

📌 Future Improvements

- User Authentication
- Database Integration
- Prediction History
- Explainable AI (SHAP/LIME)
- Batch Prediction Support
- Model Monitoring Dashboard

---

👨‍💻 Author

Amrit Kumar Gorai

B.Tech – Computer Science & Engineering (AI & Data Engineering)

VIT Vellore

Passionate about Machine Learning, Artificial Intelligence, Backend Development and Production ML Systems.