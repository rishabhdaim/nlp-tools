from nltk.tokenize import sent_tokenize


def tokenize():
    text = "God is Great! I won a lottery."
    print(sent_tokenize(text))


if __name__ == '__main__':
    tokenize()
