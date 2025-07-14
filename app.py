import streamlit as st
import pandas as pd
import re
import nltk
import pickle
from googleapiclient.discovery import build
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# Load trained model and vectorizer
with open("spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# YouTube API setup
API_KEY = "AIzaSyA3Gmt-StK8YgZMOJEFCahW6zRHlcA7W7g"  # Replace this
youtube = build("youtube", "v3", developerKey=API_KEY)

# Function to get video ID from URL
def extract_video_id(url):
    import urllib.parse as urlparse
    from urllib.parse import parse_qs
    parsed = urlparse.urlparse(url)
    if parsed.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed.path == '/watch':
            return parse_qs(parsed.query)['v'][0]
    elif parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    return None

# Function to fetch comments
def get_comments(video_id):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

    except Exception as e:
        st.error(f"Error fetching comments: {e}")
    return comments

# Streamlit App UI
st.title("🎥 YouTube Spam Comment Detector")
video_url = st.text_input("Paste YouTube Video URL:")

if st.button("Check Comments"):
    if not video_url:
        st.warning("Please enter a YouTube URL.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            with st.spinner("Fetching and analyzing comments..."):
                comments = get_comments(video_id)
                if not comments:
                    st.info("No comments found.")
                else:
                    df = pd.DataFrame(comments, columns=["Comment"])
                    df["Cleaned"] = df["Comment"].apply(preprocess)
                    vectors = vectorizer.transform(df["Cleaned"])
                    predictions = model.predict(vectors)
                    df["Prediction"] = ["🚫 Spam" if p == 1 else "✅ Not Spam" for p in predictions]
                    st.write(df[["Comment", "Prediction"]])
                


