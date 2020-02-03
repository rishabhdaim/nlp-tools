from __future__ import unicode_literals, print_function, division

import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Lang:

    def __init__(self):
        # initialize containers to hold the words and corresponding index
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        # If the word is not in the container, the word will be added to it,
        # else, update the word counter
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
    return df


# Normalize every sentence
def normalize_sentence(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace('[^\w\s]+', '')
    sentence = sentence.str.normalize('NFD')
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence


def read_sentence(df, lang1, lang2):
    sentence1 = normalize_sentence(df, lang1)
    sentence2 = normalize_sentence(df, lang2)
    return sentence1, sentence2


def process_data(destination, lang1, lang2):
    df = read_file('%s/%s-%s.txt' % (destination, lang2, lang1), lang1, lang2)
    print("Read %s sentence pairs" % len(df))
    sentence1, sentence2 = read_sentence(df, lang1, lang2)
    print("Read %s-%s sentences" % (len(sentence1), len(sentence2)))

    source = Lang()
    target = Lang()
    pairs = []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]
            source.add_sentence(sentence1[i])
            target.add_sentence(sentence2[i])
            pairs.append(full)

    print('returning source and target and pairs with size %d-%d-%d' % (len(source.word2count), len(target.word2count), len(pairs)))
    return source, target, pairs


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return input_tensor, target_tensor
