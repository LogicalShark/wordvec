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
# Names of input files in text directory, if list is empty include every file in the text directory
files = ["hollowknight_pages_current.xml"]
# Output file, will be saved in model directory
outfile = "hkmodel.bin"
# Phrases you want to keep together during tokenization
# custom_phrases = ["City of Tears", "Kingdom's Edge", "Crystal Peak"]

# To extract multi-word phrases from a file
def get_file_phrases(fname):
    phrases = []
    for line in open("list/"+fname, "r", encoding="utf-8"):
        for w in line.split(","):
            if len(word_tokenize(w)) > 1:
                phrases.append(w)
    # print("Extracted phrases:",phrases)
    return phrases

custom_phrases = get_file_phrases("hknames.txt")


class LineIterator:
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for file in self.filenames:
            # Note: custom phrases will not be condensed if split between lines
            for line in open("text/"+file, mode="r", encoding="utf-8", errors="ignore"):
                yield tokenizer.tokenize(word_tokenize(line))

if __name__ == '__main__':
    # Add custom phrases as exceptions for tokenizer
    for phrase in custom_phrases:
        tokenizer.add_mwe(word_tokenize(phrase))

    # Extract data for Word2Vec
    words = []
    if len(files) == 0:
        directory = os.fsencode("text/")
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
    words = LineIterator(files)
    for n in range(ngram-1):
        words = Phrases(words)
        phraser = Phraser(words)
    
    # Create and save model
    # 10M word vocab ~= 1 GB RAM, least frequent words are pruned
    model = Word2Vec(words, min_count=7, max_vocab_size=1000000)
    model.save("model/"+outfile)
