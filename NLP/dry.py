import nltk # pyright: ignore[reportMissingImports]
from nltk.corpus import stopwords # pyright: ignore[reportMissingImports]
from nltk.tokenize import word_tokenize # pyright: ignore[reportMissingImports]
# import pandas as pd

# Get name of all the languages
# print(stopwords.fileids())


# Lets Do This

# all_languages = stopwords.fileids()

# for lang in all_languages:
#     local_stop_words = stopwords.words(lang)
#     print(f"{lang} = ", local_stop_words[:10])

text = "I am learning Natural Language Processing in psit"

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))

custom_stop_words = stop_words.union({"student", "faculty", "psit"})

filtered_words = [ word for word in tokens if word.lower() not in custom_stop_words ]

print(filtered_words)

