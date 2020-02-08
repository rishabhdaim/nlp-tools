import urllib

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from nltk.corpus import brown
import os
print(brown.words())
print(len(brown.words()))

if "NLTK_DATA" in os.environ:
    print(os.environ.get("NLTK_DATA"))


def get_data(url):
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(strip=True)
    tokens = [t for t in text.split()]
    print(tokens)
    freq = nltk.FreqDist(tokens)
    freq.plot(20, cumulative=False, title='Work Count Token')

    stop_words = stopwords.words('english')
    clean_tokens = [t for t in tokens if t not in stop_words]
    freq = nltk.FreqDist(clean_tokens)
    freq.plot(20, cumulative=False, title='Work Count Clean Token')


if __name__ == '__main__':
    get_data('http://php.net/')