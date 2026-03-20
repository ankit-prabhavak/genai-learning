# White space tokenization
name = "Dr. Manmohan Singh"
# print(name.split())

from nltk.tokenize import WhitespaceTokenizer, word_tokenize, sent_tokenize, regexp_tokenize, TreebankWordTokenizer # pyright: ignore[reportMissingImports]
text = "NLTK is a powerful library. It is used for NLP tasks."
tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize(text)

name = "Ankit Kumar\n hello ji"
tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize(name)
# print(tokens)

# word tokenizer

words = word_tokenize(name)

# print(words)

text1 = "Don't hesitate to buy if price is $100"
words = word_tokenize(text1)
# print(words)

# Sentence tokenizer

name = "Dr. Shashi Bhusan Priyadarshini. I am king of Mathematics. I belong to Ranchi"
sent = sent_tokenize(name)
# print(sent)


# Character tokenizer
characters = "Bonjour! Abhijeet. Hallo"
# print(list(characters))

# Regexp_tokenize
vicky = "Hello JI I am rahul@example.com"
print(regexp_tokenize(vicky, pattern=r"\w+"))

# Treebank Tokenizer
# Follow Penn Treebank rules; separates contractions and punctuations properly.

micky = "Don't hestitate to ask."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(micky)

# print(tokens)

from nltk.util import ngrams # pyright: ignore[reportMissingImports]
print(ngrams(tokens, 1))

