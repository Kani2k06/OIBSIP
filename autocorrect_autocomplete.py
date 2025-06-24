# NLP Autocomplete and Autocorrect Analytics

import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ngrams, word_tokenize
from nltk.corpus import reuters
from collections import Counter, defaultdict
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('reuters')

# Step 1: Load Sample Text Data
text = reuters.raw()
tokens = word_tokenize(text.lower())
print("Total words in dataset:", len(tokens))

# Step 2: Data Cleaning
tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalpha()]
tokens = [token for token in tokens if len(token) > 1]  # remove single characters

# Step 3: Build N-gram Model for Autocomplete
def build_ngram_model(tokens, n=2):
    model = defaultdict(Counter)
    for gram in ngrams(tokens, n):
        prefix = gram[:-1]
        next_word = gram[-1]
        model[prefix][next_word] += 1
    return model

bigram_model = build_ngram_model(tokens, n=2)

# Step 4: Autocomplete Function
def autocomplete(prev_word, model, top_k=3):
    prefix = (prev_word.lower(),)
    suggestions = model.get(prefix, {})
    return [word for word, _ in suggestions.most_common(top_k)]

# Example Autocomplete Test
print("\nAutocomplete Suggestions for 'market':")
print(autocomplete('market', bigram_model))

# Step 5: Autocorrect Function using TextBlob
def autocorrect(word):
    corrected = str(TextBlob(word).correct())
    return corrected

print("\nAutocorrect Examples:")
for w in ['maket', 'transacton', 'goverment']:
    print(f"{w} → {autocorrect(w)}")

# Step 6: Word Frequency Visualization
word_freq = Counter(tokens)
common_words = word_freq.most_common(15)

words, freqs = zip(*common_words)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(freqs), y=list(words))
plt.title("Top 15 Frequent Words in Corpus")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()


