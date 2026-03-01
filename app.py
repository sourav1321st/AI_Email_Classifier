import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack

# ---------------- LOAD MODELS ----------------
# ---------------- LOAD MODELS WITH CACHING ----------------
@st.cache_resource
def load_models():
    spam_model = pickle.load(open("spam_model.pkl", "rb"))
    spam_vectorizer = pickle.load(open("spam_vectorizer.pkl", "rb"))

    cat_model = pickle.load(open("category_model.pkl", "rb"))
    cat_tfidf_word = pickle.load(open("category_tfidf_word.pkl", "rb"))
    cat_tfidf_char = pickle.load(open("category_tfidf_char.pkl", "rb"))

    urg_model = pickle.load(open("urgency_model.pkl", "rb"))
    urg_tfidf_word = pickle.load(open("urgency_tfidf_word.pkl", "rb"))
    urg_tfidf_char = pickle.load(open("urgency_tfidf_char.pkl", "rb"))
    urg_scaler = pickle.load(open("urgency_scaler.pkl", "rb"))

    return (
        spam_model, spam_vectorizer,
        cat_model, cat_tfidf_word, cat_tfidf_char,
        urg_model, urg_tfidf_word, urg_tfidf_char, urg_scaler
    )

(
    spam_model, spam_vectorizer,
    cat_model, cat_tfidf_word, cat_tfidf_char,
    urg_model, urg_tfidf_word, urg_tfidf_char, urg_scaler
) = load_models()
# ---------------- SESSION STATE FOR HISTORY ----------------
if "mail_history" not in st.session_state:
    st.session_state.mail_history = []

# ---------------- HELPER FUNCTIONS ----------------
def extract_urgency_features(text):
    text_lower = text.lower()
    urgency_words = ["urgent", "asap", "immediately", "now", "important", "priority"]
    has_urgency_word = int(any(word in text_lower for word in urgency_words))
    urgency_signal = text_lower.count("!")
    encoded_queue = 1 if "queue" in text_lower else 0
    text_length = len(text.split())
    numeric = np.array([[encoded_queue, urgency_signal, has_urgency_word, text_length]])
    return numeric

def format_spam_label(pred):
    p = str(pred).strip().lower()
    if p in {"spam", "1", "true", "yes"}:
        return "Spam"
    return "Not Spam"

def format_urgency_label(pred):
    p = str(pred).strip().lower()
    if p in {"high", "1"}:
        return "High"
    if p in {"medium", "2"}:
        return "Medium"
    if p in {"low", "0"}:
        return "Low"
    return str(pred).capitalize()

# ---------------- UI LAYOUT ----------------
st.set_page_config(page_title="AI Email Analyzer", layout="wide")
st.title("📧 AI Email Classifier Dashboard")

left, right = st.columns([2, 1])

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("Filters")

spam_filter = st.sidebar.radio(
    "Spam Filter",
    ["All", "Spam Only", "Not Spam Only"]
)

urgency_filter = st.sidebar.multiselect(
    "Urgency Level",
    ["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)
all_categories = sorted(
    {mail.get("Category", "Unknown") for mail in st.session_state.mail_history}
) if st.session_state.mail_history else []

category_filter = st.sidebar.multiselect(
    "Category Filter",
    options=all_categories,
    default=all_categories if all_categories else []
)
# ================= LEFT PANEL =================
with left:
    st.subheader("Enter Email")

    with st.form("email_form"):
        subject = st.text_input("Subject")
        body = st.text_area("Body", height=200)
        submitted = st.form_submit_button("Predict")

    if submitted:
        if subject.strip() == "" and body.strip() == "":
            st.warning("Please enter subject or body!")
        else:
            email_text = (subject + " " + body).strip()

            # Spam
            spam_vec = spam_vectorizer.transform([email_text])
            spam_raw = spam_model.predict(spam_vec)[0]
            spam_label = format_spam_label(spam_raw)

            # Category
            cat_word = cat_tfidf_word.transform([email_text])
            cat_char = cat_tfidf_char.transform([email_text])
            cat_features = hstack([cat_word, cat_char])
            cat_pred = cat_model.predict(cat_features)[0]

            # Urgency
            urg_word = urg_tfidf_word.transform([email_text])
            urg_char = urg_tfidf_char.transform([email_text])
            urg_text_vec = hstack([urg_word, urg_char])
            numeric_features = extract_urgency_features(email_text)
            numeric_scaled = urg_scaler.transform(numeric_features)
            urg_final = hstack([urg_text_vec, numeric_scaled])
            urg_raw = urg_model.predict(urg_final)[0]
            urg_label = format_urgency_label(urg_raw)

            # ---- STORE IN HISTORY ----
            preview = subject[:40] + "..." if subject else body[:40] + "..."
            st.session_state.mail_history.append({
                "Subject": subject,
                "Body": body,
                "Preview": preview,
                "Spam": spam_label,
                "Category": str(cat_pred),
                "Urgency": urg_label
            })

            # Show result
            st.subheader("📊 Prediction Result")
            if spam_label == "Spam":
                st.error("🚨 Spam Detected")
            else:
                st.success("✅ Not Spam")

            st.info(f"📂 Category: {cat_pred}")

            if urg_label == "High":
                st.error(f"⏰ Urgency: {urg_label}")
            elif urg_label == "Medium":
                st.warning(f"⚡ Urgency: {urg_label}")
            else:
                st.success(f"🟢 Urgency: {urg_label}")

# ================= RIGHT PANEL =================
with right:
    st.subheader("📜 Analyzed Mail History")

    if len(st.session_state.mail_history) == 0:
        st.info("No emails analyzed yet.")
    else:
        filtered_history = []

        for mail in st.session_state.mail_history:
            spam_value = mail["Spam"]
            urgency_value = mail["Urgency"]
            category_value = mail["Category"]

            if spam_filter == "Spam Only" and spam_value != "Spam":
                continue
            if spam_filter == "Not Spam Only" and spam_value == "Spam":
                continue
            if urgency_value not in urgency_filter:
                continue
            if category_filter and category_value not in category_filter:
                continue

            filtered_history.append(mail)

        if len(filtered_history) == 0:
            st.info("No emails match the selected filters.")
        else:
            for mail in filtered_history:
                label = f"📧 {mail['Preview']} | {mail['Spam']} | {mail['Category']} | {mail['Urgency']}"
                with st.expander(label):
                    st.markdown(f"**Subject:** {mail['Subject']}")
                    st.markdown("---")
                    st.write(mail["Body"])