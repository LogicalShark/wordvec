import os
from nltk import word_tokenize
from nltk.tokenize import word_tokenize, MWETokenizer
# For genism on Windows install Anaconda + some C compiler! WSL may work but I haven't tested on it
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser

# glove2word2vec is supposedly better, have not tried it yet
# from gensim.scripts.glove2word2vec import glove2word2vec

# 1 for unigrams, 2 for bigrams, etc.
ngram = 1

# Names of input files in text directory, if list is empty include every file in the text directory
files = ["mario_pages_current.xml"]

# Output file, will be saved in model directory
outfile = "mariomodel.bin"

# Custom multi-token phrases to be kept together, useful if you want to train with unigrams but include some bi/trigrams
custom_phrases = []  # ["Donkey Kong", "Delfino Plaza"]

# Name of file in list directory, all multi-token expressions in file will be added as custom phrases
custom_phrase_filename = "mariochars.txt"

# String replacements to make before tokenization
replacements = {"Donkey Kong": "DK", "Princess Peach": "Peach",
                "Princess Daisy": "Daisy", "King Koopa": "Bowser"}


def get_file_phrases(path):
    exp = []
    for line in open(path, "r", encoding="utf-8"):
        for w in line.split(","):
            if len(word_tokenize(w)) > 1:
                exp.append(w)
    # print("Extracted phrases:",exp)
    return exp


class LineIterator:
    def __init__(self, filenames):
        self.filenames = filenames

    def replace_all(self, line):
        for word, rep in replacements.items():
            line = line.replace(word, rep)
        return line

    def __iter__(self):
        for file in self.filenames:
            # Does not account for multi-word expressions split between lines
            for line in open("text/"+file, mode="r", encoding="utf-8", errors="ignore"):
                yield tokenizer.tokenize(word_tokenize(self.replace_all(line)))


if __name__ == '__main__':
    tokenizer = MWETokenizer(separator=" ")
    # Extract MWEs from custom_phrase_filename
    if len(custom_phrase_filename) > 0:
        custom_phrases += get_file_phrases("list/"+custom_phrase_filename)
    # Add custom phrases as exceptions for tokenizer
    for phrase in custom_phrases:
        tokenizer.add_mwe(word_tokenize(phrase))

    # Create tokens for Word2Vec
    words = []
    if len(files) == 0:
        directory = os.fsencode("text/")
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
    words = LineIterator(files)

    # Turn into bigrams/trigrams etc.
    for n in range(ngram-1):
        words = Phrases(words)
        phraser = Phraser(words)

    # Create and save model, 10M word vocab = 1 GB RAM
    model = Word2Vec(words, min_count=7, max_vocab_size=1000000)
    model.save("model/"+outfile)
