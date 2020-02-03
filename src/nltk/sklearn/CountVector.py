from sklearn.feature_extraction.text import CountVectorizer


def count_vector():

    vectorizer = CountVectorizer()
    data_corpus = ["guru99 is the best site for online tutorials. I love to visit guru99."]
    vocabulary = vectorizer.fit(data_corpus)
    x = vectorizer.transform(data_corpus)
    print(x.toarray())
    print(vocabulary.get_feature_names())


if __name__ == '__main__':
    count_vector()
