from nltk.corpus import wordnet


def synonyms(word):
    syns = wordnet.synsets(word)
    print(str(set(syns)) + " " + str(len(syns)))


def synonym_and_antonym(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print(set(synonyms))
    print(set(antonyms))


if __name__ == '__main__':
    word = 'dog'
    synonyms(word)
    synonym_and_antonym('active')

