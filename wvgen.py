import os
from nltk import word_tokenize
from nltk.tokenize import word_tokenize, MWETokenizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
# glove2word2vec is supposedly faster, I haven't tried it yet
from gensim.scripts.glove2word2vec import glove2word2vec
tokenizer = MWETokenizer(separator=" ")

# 1 for unigrams, 2 for bigrams, etc.
ngram = 1
# Names of input files, if empty list will use every file in the text directory
files = ["hollowknight_pages_current.xml"]
# Output file, will be saved in model/
outfile = "hkmodel.bin"
# Phrases you want to keep from being separated during tokenization
custom_phrases = ["City of Tears", "Kingdom's Edge", "Crystal Peak", "Howling Cliffs", "Queen's Gardens", "Fungal Wastes", "Fog Canyon", "White Palace", "Royal Waterways", "The Hive", "The Abyss", "Forgotten Crossroads", "Resting Grounds"]

class LineIterator:
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for file in self.filenames:
            for line in open("text/"+file, mode="r", encoding="utf-8", errors="ignore"):
                yield tokenizer.tokenize(word_tokenize(line))

def addExceptions():
    for phrase in custom_phrases:
        tokenizer.add_mwe(phrase.split())

if __name__ == '__main__':
    addExceptions()
    words = []
    if len(files) == 0:
        directory = os.fsencode("text/")
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
    words = LineIterator(files)
    for n in range(ngram-1):
        words = Phrases(words)
        phraser = Phraser(words)
    # 10M word vocab ~= 1 GB RAM, if vocab limit exceeded least frequent words are pruned
    model = Word2Vec(words, min_count=5, max_vocab_size=1000000)
    model.save("model/"+outfile)
