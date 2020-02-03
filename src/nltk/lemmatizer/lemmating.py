from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from collections import defaultdict


def lemmatize(words):
    wordnet_lemma = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for w in words:
        print('Lemma for %s is %s AND Stemming is %s' % (w, wordnet_lemma.lemmatize(w), stemmer.stem(w)))


def lemmatize_pos(words, tag_map):
    wordnet_lemma = WordNetLemmatizer()
    for token, tag in pos_tag(words):
        lemma = wordnet_lemma.lemmatize(token, tag_map[tag[0]])
        print('%s => %s' % (token, lemma))


if __name__ == '__main__':
    text = "studies studying cries cry"
    tokens = word_tokenize(text)
    lemmatize(tokens)
    print('-------------------------------')
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    text = "guru99 is a totally new kind of learning experience."
    tokens = word_tokenize(text)
    lemmatize_pos(tokens, tag_map)
