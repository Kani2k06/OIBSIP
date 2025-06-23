import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud

# Download stopwords
nltk.download('stopwords')

# Sample dataset
data = {
    'text': [
        'I love this app, it’s amazing!',
        'Worst app ever. Crashes every time.',
        'Just okay, not good not bad.',
        'Fantastic user interface and smooth.',
        'Waste of time. Too many ads.',
        'Very helpful and easy to use.',
        'Bad experience. Didn’t like it at all.',
        'Great features and performance!',
        'Not worth downloading.',
        'Average app. Could be better.'
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Text cleaning function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# WordClouds
for sentiment in df['sentiment'].unique():
    words = ' '.join(df[df['sentiment'] == sentiment]['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {sentiment} sentiment")
    plt.show()
