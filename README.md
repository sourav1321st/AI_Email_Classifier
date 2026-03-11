link : https://aiemailclassifier-bysrv.streamlit.app/
# AI Email Classifier

## Overview

AI Email Classifier is a machine learning based web application that analyzes email content and automatically predicts important attributes of the email. The system helps users quickly understand whether an email is spam, what category it belongs to, and how urgent it is.

The application uses Natural Language Processing (NLP) techniques and machine learning models to process the email text and provide intelligent predictions through an interactive web dashboard built using Streamlit.

This project demonstrates the practical implementation of text classification, feature engineering, and model deployment.

---

# Features

• Spam Detection – Identifies whether an email is spam or legitimate
• Email Category Classification – Predicts the type of email (such as support, work, or general communication)
• Urgency Detection – Determines whether an email is High, Medium, or Low priority
• Interactive Dashboard – Users can input email subject and body through a web interface
• Email History Tracking – Keeps track of analyzed emails in the session
• Filter Options – Allows filtering results based on spam status, urgency, and category

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

### Web Application

Streamlit

### Model Storage

Pickle

### Development Tools

Jupyter Notebook

---

# Project Structure

```
AI_Email_Classifier
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
│
├── DataSets_
│   ├── SpamDataset.csv
│   └── mergeSet.csv
│
├── Model
│   ├── spam_model.ipynb
│   ├── category_model.ipynb
│   └── urgency_model.ipynb
│
├── Trained Models
│   ├── spam_model.pkl
│   ├── spam_vectorizer.pkl
│   ├── category_model.pkl
│   ├── category_tfidf_word.pkl
│   ├── category_tfidf_char.pkl
│   ├── urgency_model.pkl
│   ├── urgency_tfidf_word.pkl
│   ├── urgency_tfidf_char.pkl
│   └── urgency_scaler.pkl
```

---

# How It Works

The system processes email text and performs three types of classification.

1. The user enters the email subject and body in the Streamlit dashboard.
2. The text is preprocessed and converted into numerical features using TF-IDF vectorization.
3. The processed features are passed to three trained machine learning models:

   * Spam Detection Model
   * Email Category Classification Model
   * Urgency Prediction Model
4. The predictions are displayed on the dashboard.

---

# Machine Learning Models

### Spam Detection Model

This model classifies emails as spam or non-spam using TF-IDF features extracted from the email text.

### Category Classification Model

This model predicts the category of the email using both word-level and character-level TF-IDF features.

### Urgency Prediction Model

This model determines the urgency level of the email by combining:

* TF-IDF text features
* Custom numeric features such as urgency keywords, punctuation signals, and text length.

---

# Datasets

The project uses two datasets for training the models.

### Spam Dataset

Used for training the spam detection model.

### Merged Email Dataset

Used for training category classification and urgency detection models.

---

# Installation

### 1 Clone the repository

```
git clone https://github.com/yourusername/AI_Email_Classifier.git
cd AI_Email_Classifier
```

### 2 Install dependencies

```
pip install -r requirements.txt
```

### 3 Run the application

```
streamlit run app.py
```

---

# Application Demo

After running the application, open your browser and navigate to:

```
http://localhost:8501
```

You can then enter an email subject and body to analyze the email.

---

# Example Input

Subject

```
Urgent: Server is down
```

Body

```
Please fix the server immediately. The system is not responding.
```

Output

Spam Status → Not Spam
Category → Technical Support
Urgency → High

---

# Future Improvements

• Integration with real email services (Gmail API, Outlook API)
• Deep learning models such as LSTM or Transformers
• Larger datasets for improved accuracy
• Email attachment analysis
• Automatic email sorting and tagging

---

# Author

Sourav Meher
Final Year B.Tech Student

---

# License

This project is developed for educational and learning purposes.
