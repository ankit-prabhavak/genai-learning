from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

corpus = [
    'This is the first document',
    'This document is the second document',
    'and this is the third one',
    'is this the first document',
]

vect = CountVectorizer().fit(corpus)
# print(vect.get_feature_names_out())

# print(vect.transform(corpus).toarray())




# TF-IDF
x = vect.fit_transform(corpus)

tfidf = TfidfTransformer()
x_tfidf = tfidf.fit_transform(X=x)

print(x_tfidf.toarray())
print(vect.get_feature_names_out())


