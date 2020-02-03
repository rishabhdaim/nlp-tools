from nltk.corpus import brown
import os
print(brown.words())
print(len(brown.words()))

if "NLTK_DATA" in os.environ:
    print(os.environ.get("NLTK_DATA"))