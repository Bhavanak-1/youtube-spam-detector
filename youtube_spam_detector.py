import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

print("✅ Starting script...")

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("youtube_spam.csv")

# Show actual column names
print("🧾 Columns in dataset:", df.columns.tolist())

# Rename them to match expected names
df.rename(columns={"comment": "content", "class": "label"}, inplace=True)

# Now it's safe to drop missing values
df.dropna(subset=['content', 'label'], inplace=True)


# Encode labels
# Fix column names to match your dataset
df.rename(columns={"comment": "content", "class": "label"}, inplace=True)

# Drop rows with unexpected or missing labels
df = df[df['label'].isin(['spam', 'ham'])]

# Encode labels: spam=1, ham=0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # 'spam'=1, 'ham'=0

# Preprocess function
def preprocess(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    words = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
print("🧹 Cleaning text...")
df['cleaned'] = df['content'].apply(preprocess)

# TF-IDF vectorization
print("🔠 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['label']

# Split data
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🤖 Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
print("📊 Evaluating model...")
y_pred = model.predict(X_test)

print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predict function
def predict_spam(comment):
    comment_clean = preprocess(comment)
    vec = vectorizer.transform([comment_clean]).toarray()
    prediction = model.predict(vec)[0]
    label = "Spam" if prediction == 1 else "Not Spam"
    return label

# Example
print("\n🔍 Example prediction:")
example = "Check out my channel for free giveaways!!"
print(f"Comment: {example}")
print(f"Prediction: {predict_spam(example)}")

