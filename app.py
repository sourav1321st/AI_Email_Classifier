import streamlit as st # type: ignore
import pickle
import re
import pandas as pd

# -------------------------------
# Load Models
# -------------------------------
spam_model = pickle.load(open("spam_model.pkl", "rb"))
spam_vectorizer = pickle.load(open("spam_vectorizer.pkl", "rb"))
category_model = pickle.load(open("category_model.pkl", "rb"))
urgency_model = pickle.load(open("urgency_model.pkl", "rb"))

# -------------------------------
# Tag Function
# -------------------------------
def tag(text, color):
    return f"""
        <span style="
            background-color:{color};
            color:white;
            padding:4px 10px;
            border-radius:8px;
            font-size:14px;
            margin-right:6px;">
            {text}
        </span>
    """

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# Prediction Function
# -------------------------------
def predict_email(subject, body):
    full_text = subject + " " + body
    text_clean = clean_text(full_text)

    # Spam prediction
    spam_vec = spam_vectorizer.transform([text_clean])
    spam_pred = spam_model.predict(spam_vec)[0]
    spam_label = "Spam" if spam_pred == 1 else "Not Spam"

    # Category prediction
    category = category_model.predict([text_clean])[0]

    # Urgency prediction
    urgency = urgency_model.predict([text_clean])[0]

    return spam_label, category, urgency

# -------------------------------
# Session State Initialization
# -------------------------------
if "emails" not in st.session_state:
    st.session_state.emails = []

st.title("AI Email Classifier Dashboard")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

spam_filter = st.sidebar.radio(
    "Spam Filter",
    ["All", "Spam Only", "Not Spam Only"]
)

urgency_filters = st.sidebar.multiselect(
    "Urgency Level",
    ["high", "medium", "low"],
    default=["high", "medium", "low"]
)

# -------------------------------
# Email Input Form
# -------------------------------
st.subheader("Enter Email")

with st.form("email_form", clear_on_submit=True):
    subject = st.text_input("Subject")
    body = st.text_area("Body")
    submitted = st.form_submit_button("Predict")

if submitted:
    if subject.strip() and body.strip():
        spam, category, urgency = predict_email(subject, body)

        st.session_state.emails.append({
            "Subject": subject,
            "Body": body,
            "Spam": spam,
            "Category": category,
            "Urgency": urgency
        })

        st.rerun()
    else:
        st.warning("Please enter both subject and body")

# -------------------------------
# Display Emails Table
# -------------------------------
if st.session_state.emails:
    df = pd.DataFrame(st.session_state.emails)

    # Apply Spam Filter
    if spam_filter == "Spam Only":
        df = df[df["Spam"] == "Spam"]
    elif spam_filter == "Not Spam Only":
        df = df[df["Spam"] == "Not Spam"]

    # Apply Urgency Filter
    df = df[df["Urgency"].str.lower().isin(urgency_filters)]

    st.subheader("Email List")

    if not df.empty:
        display_df = df[["Subject", "Spam", "Category", "Urgency"]]
        st.dataframe(display_df, use_container_width=True)

        # Select email to view full details
        st.subheader("Click to view email")

        options = ["None"] + list(df.index)

        selected_index = st.selectbox(
            "Select an email",
            options,
            format_func=lambda x: "Select email..." if x == "None" else df.loc[x, "Subject"]
        )

        if selected_index != "None":
            st.subheader("Email Details")
            subject = df.loc[selected_index, "Subject"]
            body = df.loc[selected_index, "Body"]
            spam = df.loc[selected_index, "Spam"]
            category = df.loc[selected_index, "Category"]
            urgency = df.loc[selected_index, "Urgency"]

            st.markdown(f"**Subject:** {subject}")
            st.markdown(f"**Body:** {body}")

            # Color logic
            spam_color = "#e74c3c" if spam == "Spam" else "#2ecc71"

            urgency_colors = {
                "high": "#e74c3c",
                "medium": "#f39c12",
                "low": "#3498db"
            }

            urgency_color = urgency_colors.get(urgency.lower(), "#7f8c8d")

            # Display tags
            st.markdown(
              tag(spam, spam_color) +
              tag(category, "#6c5ce7") +
              tag(urgency.capitalize(), urgency_color),
              unsafe_allow_html=True
            )

    else:
        st.info("No emails match selected filters.")
