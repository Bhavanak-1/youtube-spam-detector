import pandas as pd
import re
import string
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

# Load dataset
df = pd.read_csv("youtube_spam_dataset.csv")  # Make sure this file exists

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
df["cleaned"] = df["CONTENT"].apply(preprocess)  # Use actual column name here

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["CLASS"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save the model and vectorizer
with open("spam_classifier_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully.")
# Save model
with open("spam_classifier_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

