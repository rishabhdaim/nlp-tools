import gensim
from nltk.corpus import abc


def similar_words(word):
    model = gensim.models.Word2Vec(abc.sents())
    x = list(model.wv.vocab)
    data = model.wv.most_similar(word)
    print(data)


if __name__ == '__main__':
    similar_words('science')
    similar_words('love')
