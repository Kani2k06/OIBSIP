# Google Play Store Data Analysis

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
df = pd.read_csv("apps.csv")  
print(df.head())
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = df[df['Category'] != '1.9']
df['Reviews'] = df['Reviews'].astype(int)
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)
df['Price'] = df['Price'].str.replace('$', '').astype(float)
df['Size'] = df['Size'].replace('Varies with device', np.nan)
def convert_size(size):
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'k' in size:
            return float(size.replace('k', '')) / 1024
    return np.nan  # for non-strings or unrecognized formats

df['Size'] = df['Size'].apply(convert_size)
df['Size'].fillna(df['Size'].mean(), inplace=True)

print("\nCleaned Data Types:\n", df.dtypes)
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='bar')
plt.title("App Count per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
top_rated = df.sort_values(by='Rating', ascending=False).head(10)
print("\nTop 10 Rated Apps:\n", top_rated[['App', 'Category', 'Rating']])
popular_apps = df.sort_values(by='Installs', ascending=False).head(10)
print("\nTop 10 Installed Apps:\n", popular_apps[['App', 'Installs']])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Size', y='Rating', data=df, hue='Category', alpha=0.6)
plt.title("App Size vs Rating")
plt.legend([],[], frameon=False)
plt.show()
sample_reviews = [
    "Great app, loved the UI!",
    "Worst experience ever. It crashes always.",
    "Good but has some bugs.",
    "Very helpful for learning."
]
print("\nSentiment Analysis Sample:")
for review in sample_reviews:
    polarity = TextBlob(review).sentiment.polarity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    print(f"Review: {review}\n → Sentiment: {sentiment}\n")
text = " ".join(app for app in df['App'])
wordcloud = WordCloud(background_color="white").generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Popular App Words")
plt.show()
