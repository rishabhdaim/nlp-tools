from collections import Counter
import nltk


def counter(text):
    lower_text = text.lower()
    tokens = nltk.word_tokenize(lower_text)
    tags = nltk.pos_tag(tokens)
    counts = Counter(tag for word, tag in tags)
    print(counts)


def freq_dist(text):
    words = nltk.word_tokenize(text)
    fd = nltk.FreqDist(words)
    # fd.plot()
    print(fd.freq('and'))
    print(fd.freq('the'))
    print(fd.freq('Java'))


def bigrams(text):
    tokens = nltk.word_tokenize(text)
    print(list(nltk.bigrams(tokens)))
    print(list(nltk.trigrams(tokens)))


if __name__ == '__main__':
    text = 'Guru99 is one of the best sites to learn WEB, SAP, Ethical Hacking and much more online.'
    counter(text)
    text = 'Guru99 is the site where you can find the best tutorials for Software Testing     Tutorial, SAP Course for ' \
           'Beginners. Java Tutorial for Beginners and much more. Please     visit the site guru99.com and much more.'
    freq_dist(text)
    bigrams(text)


