link : https://aiemailclassifier-bysrv.streamlit.app/
# AI Email Classifier

## Overview

AI Email Classifier is a Machine Learning and Natural Language Processing based web application that analyzes email content and predicts important characteristics of the email automatically.

The system classifies emails into three aspects:

• Spam Detection
• Email Category Classification
• Email Urgency Prediction

The application is deployed using **Streamlit**, allowing users to enter an email subject and body and instantly receive intelligent predictions.

This project demonstrates the practical implementation of **text classification, feature engineering, and machine learning model deployment**.

---

# Features

• Detects whether an email is **Spam or Not Spam**
• Predicts the **category of the email**
• Determines **urgency level (High / Medium / Low)**
• Interactive **Streamlit web dashboard**
• Uses **NLP and Machine Learning models**
• Real-time email analysis

---

# Tech Stack

### Programming Language

Python

### Machine Learning & NLP

Scikit-learn
TF-IDF Vectorizer
NumPy
SciPy
Pandas

### Web Framework

Streamlit

### Model Storage

Pickle (.pkl)

### Development Tools

Jupyter Notebook
VS Code

---

# Project Structure

```
AI_EMAIL_CLASSIFIER
│
├── .devcontainer
│
├── AgileDocumentation
│   ├── Agile_Template_v0.1.xls
│   ├── Defect_Tracker_Template_v0.1.xlsx
│   └── Unit_Test_Plan_v0.1.xlsx
│
├── DataSets_
│   ├── mergeSet.csv
│   └── SpamDataset.csv
│
├── Model
│   ├── category_model.ipynb
│   ├── spam_model.ipynb
│   └── urgency_model.ipynb
│
├── StreamlitDeployment
│   └── app.py
│
├── category_model.pkl
├── category_tfidf_char.pkl
├── category_tfidf_word.pkl
│
├── spam_model.pkl
├── spam_vectorizer.pkl
│
├── urgency_model.pkl
├── urgency_scaler.pkl
├── urgency_tfidf_char.pkl
├── urgency_tfidf_word.pkl
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# How the System Works

1. The user enters the **email subject and body** in the Streamlit web interface.
2. The subject and body are combined into a single text input.
3. The text is transformed into numerical features using **TF-IDF Vectorization**.
4. These features are passed into three trained machine learning models:

   * Spam Detection Model
   * Email Category Classification Model
   * Urgency Prediction Model
5. The predictions are displayed instantly on the dashboard.

---

# Machine Learning Models

## Spam Detection Model

This model predicts whether an email is spam or legitimate.

Files used:

* spam_model.pkl
* spam_vectorizer.pkl

Technique used:
TF-IDF Vectorization + Machine Learning Classification

---

## Email Category Classification

This model predicts the category of the email.

Files used:

* category_model.pkl
* category_tfidf_word.pkl
* category_tfidf_char.pkl

Features used:
• Word-level TF-IDF features
• Character-level TF-IDF features

---

## Urgency Prediction Model

This model determines the urgency level of the email.

Files used:

* urgency_model.pkl
* urgency_scaler.pkl
* urgency_tfidf_word.pkl
* urgency_tfidf_char.pkl

Features used:
• Text-based TF-IDF features
• Custom urgency indicators

---

# Datasets

The datasets used for training the models are stored in the **DataSets_** directory.

### SpamDataset.csv

Used for training the spam detection model.

### mergeSet.csv

Used for training the category classification and urgency prediction models.

---

# Model Development

The machine learning models were trained using Jupyter notebooks located in the **Model** directory:

• spam_model.ipynb
• category_model.ipynb
• urgency_model.ipynb

These notebooks contain the complete pipeline for:

• Data preprocessing
• Feature extraction
• Model training
• Model evaluation
• Model export (.pkl files)

---

# Installation

## Clone the repository

```
git clone https://github.com/sourav1321st/AI_Email_Classifier.git
cd AI_EMAIL_CLASSIFIER
```

---

## Install dependencies

```
pip install -r requirements.txt
```

---

## Run the application

```
streamlit run StreamlitDeployment/app.py
```

---

# Running the Application

After running the application, open your browser and visit:

```
http://localhost:8501
```

You can then input an email subject and body to analyze the email.

---

# Example

Input Email

Subject:

```
Urgent: Server is down
```

Body:

```
Please fix the server immediately. The system is not responding.
```

Output:

Spam Status → Not Spam
Category → Technical Support
Urgency → High

---

# Agile Documentation

The **AgileDocumentation** folder contains project management documents such as:

• Agile project template
• Defect tracking sheet
• Unit testing plan

These documents help track development progress and testing activities.

---

# Future Improvements

• Integration with real email services (Gmail API / Outlook API)
• Deep learning based models (LSTM / Transformers)
• Larger datasets for better accuracy
• Automatic email classification for inbox management
• Real-time email monitoring system

---

# Author

Sourav Meher
Final Year B.Tech Student

---

# License

This project is created for educational and learning purposes.
