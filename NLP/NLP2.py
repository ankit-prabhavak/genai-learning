import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('brown')
# print("NLP libraries downloaded successfully!")


# nltk.download('stopwords')

from nltk.corpus import brown
# print("Number of Categories in Brown Corpus:", len(brown.categories()))
# print("Categories in Brown Corpus:", brown.categories())


#  Get first 10 words from the 'news' category
# print("Number of Words in Brown Corpus:", len(brown.words(categories='news')[:])) # brown has over 1 million words in total, so we are just printing the first 10 for demonstration
# print("First 10 Words in 'news' Category:", brown.words(categories='news')[:10])

# Print the 10 sample words from each category
# for item in brown.categories():
#     print(f"First 10 words {item}", brown.words(categories=item)[:10])

from nltk.tokenize import word_tokenize

# Corpus, Vocabulary, Tokens, Words


# Choose a category from Brown corpus and print row string

# category = "news"
# raw_text = brown.raw(categories=category) 
# raw_tokens  = word_tokenize(raw_text)
# print(f"Total raw_tokens in '{category}': {len(raw_tokens)}")

# words: all words in the category

# Tokenized words + punctuation
# words = brown.words(categories=category) 
# print(f"Total words in '{category}': {len(words)}")
# print(words)


# Print punctuations only 
# punctuations = [w for w in words if not w.isalnum() ]
# print("Total punctuations are: ", len(punctuations))
# print(punctuations)


# Lets do the same using our own corpus
from nltk.tokenize import word_tokenize

# corpus = [
#     'This is the first. document',
#     'This document 10 is the second! document',
#     'and this 12 is the third one,',
#     'is this the first@ document',
# ]

# raw_text = " ".join(corpus)   # join your own documents
# raw_tokens = word_tokenize(raw_text)

# print("Total number of Tokens:", len(raw_tokens))
# print(raw_tokens)
# print("\n")
# punctuations = [w for w in raw_tokens if not w.isalnum() ]
# print("Total punctuations are: ", len(punctuations))
# print(punctuations)
# print("\n")
# non_punctuations = [w for w in raw_tokens if w.isalnum() ]
# print("Total punctuations are: ", len(non_punctuations))
# print(non_punctuations)
# print("\n")
# words_only = [w for w in raw_tokens if w.isalpha() ]
# print("Total punctuations are: ", len(words_only))
# print(words_only)
# print("\n")
# paragraph = ' '.join(words_only)
# print(paragraph)

# print("\nTotal unique words in paragraph: ", len(set(words_only)))



from matplotlib import pyplot as plt
# Plot Total words vs vocab size per brown corpus category

cat_list = brown.categories()

words_list = []
vocab = []

for cat in cat_list:
    words = brown.words(categories=cat) 
    words_only = [w for w in words if w.isalpha() ]
    words_list.append(len(words_only))
    vocab.append(len(set(words_only)))


# Positions for bars
x = range(len(cat_list))
width = 0.4

plt.figure(figsize=(14, 7))

# Total words bars
plt.bar([i - width/2 for i in x], words_list, width=width, label="Total Words", color="steelblue")

# Vocabulary size bars
plt.bar([i + width/2 for i in x], vocab, width=width, label="Vocabulary Size", color="orange")

plt.xticks(list(x), cat_list, rotation=45, ha="right")
plt.ylabel("Count")
plt.title("Brown Corpus: Total Words vs Vocabulary Size per Category")
plt.legend()
plt.tight_layout()
plt.show()