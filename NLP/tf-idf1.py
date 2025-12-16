from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [

    "I love Python and AI",
    "AI will shape the future"
]

vectorizer=TfidfVectorizer()

tfidf_matrix =vectorizer.fit_transform(corpus)

print("\nidf values: ")
for ele1,ele2 in zip(vectorizer.get_feature_names_out(),vectorizer.idf_):
    print(ele1,":",ele2)

print("\nFeature names: ")
print(vectorizer.get_feature_names_out())

print("\nWord indexes: ")
print(vectorizer.vocabulary_)
print('\ntf-idf value:')
print(tfidf_matrix)
print('\ntf-idf values in matrix forms: ')
print(tfidf_matrix.toarray())