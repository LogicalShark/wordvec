from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
# For doing arithmetic with word vectors
# e.g. king + woman - man ~= queen, Paris + Italy - France ~= Rome

# Name of model file, output of wvgen.py
modelf = "mariomodel.bin"
# Words to add
positives = ["City of Tears", "Howling Cliffs"]
# Words to subtract
negatives = ["Greenpath"]
# Number of results
n_results = 3

def equationResult(model):
    try:
        # Handle multi-word expressions
        p = [' '.join(word_tokenize(x)) for x in positives]
        n = [' '.join(word_tokenize(x)) for x in negatives]
        print(model.wv.most_similar_cosmul(
            positive=p, negative=n, topn=n_results))
    except KeyError:
        print("Error: input word not in vocabulary\n")

def simWord(model, word):
    try:
        print(model.wv.similar_by_word(word, topn=n_results))
    except KeyError:
        print("Error: input word not in vocabulary\n")

if __name__ == "__main__":
    model = Word2Vec.load("model/"+modelf)
    if len(positives) == 1 && len(negatives) == 0:
        simWord(model, positives[0])
    equationResult(model)
