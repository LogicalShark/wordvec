import os
from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.scripts.glove2word2vec import glove2word2vec

# 1 for unigrams, 2 for bigrams, etc.
ngram = 1
# Paths to input files, if empty it will use every file in the text directory
files = ["text/hollowknight_pages_current.xml"]
# Output file
outfile = "model/hkmodel.bin"


def flatten(l): return [item for sublist in l for item in sublist]

class SentenceIterator:
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __iter__(self):
        for path in self.filepaths:
            for line in open(path, encoding="utf-8", errors="ignore"):
                yield word_tokenize(line)


if __name__ == '__main__':
    sentences = []
    if len(files) == 0:
        directory = os.fsencode("text/")
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
    sentences = SentenceIterator(files)
    s = sentences
    for n in range(ngram-1, 0, -1):
        phrases = Phrases(s)
        phraser = Phraser(phrases)
    model = Word2Vec(sentences, min_count=5, max_vocab_size=1000000) # 10M word vocab ~= 1 GB RAM, prunes least frequent words
    model.save(outfile)
