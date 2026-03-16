# Sample customer reviews
reviews = [
    "The product quality is excellent",
    "Delivery was late and the product is bad",
    "Average experience, nothing special",
    "I am very happy with the fast delivery",
    "The product is poor and disappointing"
]

# Corresponding labels (for reference)
labels = ["positive", "negative", "neutral", "positive", "negative"]
import re

def preprocess_text(text):
    text = text.lower()                       # Lowercasing
    text = re.sub(r'[^a-z\s]', '', text)      # Remove punctuation
    return text

# Apply preprocessing
cleaned_reviews = [preprocess_text(review) for review in reviews]

print("Cleaned Reviews:")

for review in cleaned_reviews:
    print(review)

from sklearn.preprocessing import OneHotEncoder

import numpy as np
# Tokenize words
tokenized_reviews = [review.split() for review in cleaned_reviews]
# Flatten word list
all_words = np.array(sum(tokenized_reviews, [])).reshape(-1, 1)
# One Hot Encoder
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoded_words = onehot_encoder.fit_transform(all_words)
# Vocabulary
vocab = onehot_encoder.get_feature_names_out()

print("\nVocabulary:")
print(vocab)

print("\nOne-Hot Encoded Word Vectors:")
print(onehot_encoded_words)

from sklearn.feature_extraction.text import CountVectorizer

# Initialize BoW vectorizer
vectorizer = CountVectorizer()
# Fit and transform the cleaned reviews
bow_matrix = vectorizer.fit_transform(cleaned_reviews)

# Convert to array for display
bow_array = bow_matrix.toarray()
# Feature names
feature_names = vectorizer.get_feature_names_out()
print("\nBag of Words Vocabulary:")
print(feature_names)

print("\nBag of Words Representation:")
print(bow_array)
