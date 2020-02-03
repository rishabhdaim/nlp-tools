from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def stemmer(words):

    ps = PorterStemmer()
    root_words = []
    for word in words:
        root_words.append(ps.stem(word))
    print(root_words)


if __name__ == '__main__':
    words = ['wait', 'waiting', 'waited', 'waits']
    stemmer(words)
    print('------------------------------')
    sentence = 'Hello Guru99, You have to build a very good site and I love visiting your site.'
    words = word_tokenize(sentence)
    stemmer(words)

