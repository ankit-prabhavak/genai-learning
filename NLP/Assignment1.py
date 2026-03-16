import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# Sample Feedback
feedbacks = [
    "The faculty was very supportive and the lectures were really helpful.",
    "I am not satisfied with the lab sessions.",
    "The course content is good.",
    "Lan were not working properly."
]


for feedback in feedbacks:
    # tokenize full string
    tokens = word_tokenize(feedback)
    

    # remove punctuation tokens
    tokens = [t for t in tokens if t.isalpha()]

    # print(tokens)

    stop_words = set(stopwords.words('english'))
    
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    print(filtered_words)
