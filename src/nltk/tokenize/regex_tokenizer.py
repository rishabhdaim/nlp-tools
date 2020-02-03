from nltk.tokenize import RegexpTokenizer


def tokenize():

    tokenizer = RegexpTokenizer(r'\w+')

    filterd_text = tokenizer.tokenize('Hello Guru99, You have build a very good site and I love visiting your site.')

    print(filterd_text)


if __name__ == '__main__':
    tokenize();