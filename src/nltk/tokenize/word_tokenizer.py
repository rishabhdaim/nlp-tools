from nltk.tokenize import word_tokenize


def tokenize():
    text = "God is Great! I won a lottery."
    print(word_tokenize(text))


if __name__ == '__main__':
    tokenize()

