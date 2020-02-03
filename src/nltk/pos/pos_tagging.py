from nltk import pos_tag
from nltk import RegexpParser
import nltk
import time


def pos_chunks():
    text = "learn php from guru99 and make study easy".split()
    print("After Split: ", text)
    tokens_tag = pos_tag(text)
    print("After Token: ", tokens_tag)
    patterns = """mychunk: {<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
    chunker = RegexpParser(patterns)
    print("After Regex: ", chunker)
    output = chunker.parse(tokens_tag)
    print("After Chunking: ", output)


def entity_detection(text):
    tokens = nltk.word_tokenize(text)
    print(tokens)
    tags = nltk.pos_tag(tokens)
    print(tags)
    grammer = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammer)
    result = cp.parse(tags)
    print(result)
    result.draw()


if __name__ == '__main__':
    pos_chunks()
    time.sleep(1)
    print('-----------------------------')
    text = 'Temperature of New York.'
    entity_detection(text)
    print('-----------------------------')
    text = 'learn php from guru99'
    entity_detection(text)

