import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import datetime

# Download NLTK stopwords
nltk.download('stopwords')

# Text preprocessing function
def preprocess(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())  # Remove non-alphabets
    words = text.split()
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Load and train the model
@st.cache_resource
def train_model():
    df = pd.read_csv("youtube_spam_dataset.csv")
    df = df.rename(columns={"CLASS": "label", "CONTENT": "comment"})
    df = df.dropna(subset=["comment", "label"])
    
    # Preprocess the comments
    df['cleaned'] = df['comment'].apply(preprocess)

    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['cleaned']).toarray()
    y = df['label']

    # Train the model
    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer

# Train the model once and reuse
model, vectorizer = train_model()

# App UI
st.title("📺 YouTube Spam Comment Detector")
st.write("Enter a YouTube comment below to check if it's spam:")

user_input = st.text_area("💬 Your Comment", "")

# Predict button
if st.button("🚀 Predict"):
    cleaned = preprocess(user_input)
    vec = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vec)[0]

    # Log prediction
    def log_prediction(comment, prediction):
        with open("prediction_log.txt", "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            label = "Spam" if prediction == 1 else "Not Spam"
            f.write(f"{timestamp} - INPUT: {comment} - RESULT: {label}\n")

    log_prediction(user_input, prediction)

    # Show result
    if prediction == 1:
        st.error("🛑 Spam Comment Detected!")
    else:
        st.success("✅ Not Spam")

# CSV Upload section
st.subheader("📂 Upload CSV for Batch Detection")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'comment' not in data.columns:
        st.error("❌ CSV must have a 'comment' column.")
    else:
        data = data.dropna(subset=["comment"])
        data['cleaned'] = data['comment'].apply(preprocess)
        X_batch = vectorizer.transform(data['cleaned']).toarray()
        data['prediction'] = model.predict(X_batch)
        data['prediction_class'] = data['prediction'].apply(lambda x: "Spam" if x == 1 else "Not Spam")
        st.write("✅ Predictions:")
        st.dataframe(data[['comment', 'prediction_class']])
        st.download_button("📥 Download Results", data.to_csv(index=False), "predictions.csv", "text/csv")




