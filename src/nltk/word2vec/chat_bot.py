import json
import string

import pandas
from nltk.corpus import stopwords
from textblob import Word
from gensim.models import Word2Vec


def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data


def convert_data(data, language):
    df = pandas.DataFrame(data)
    stop_words = stopwords.words(language)
    df['patterns'] = df['patterns'].apply(', '.join)
    df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
    df['patterns'] = df['patterns'].str.replace('[^\w\s]', '')
    df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if not x.isdigit()))
    df['patterns'] = df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df


def create_model(df, destination):
    bigger_list = []
    for i in df['patterns']:
        li = list(i.split())
        bigger_list.append(li)
    print("Data format for the overall list:", bigger_list)

    model = Word2Vec(bigger_list, min_count=1, size=300, workers=4)

    model.save(destination + '/words.model')
    model.save(destination + '/words.bin')


def test_model(source):
    model = Word2Vec.load(source + '/words.bin')
    print('Similar to thanks ', model.wv.most_similar('thanks'))
    print('Doesn\'t match', model.wv.doesnt_match('See you later, thanks for visiting'.split()))
    print('Similarity between please and see', model.wv.similarity('please', 'see'))
    print('Similar to Kind', model.wv.similar_by_word('kind'))


if __name__ == '__main__':
    create_model(convert_data(load_data('resources/intent.json'), 'english'), 'target/model')
    test_model('target/model')
