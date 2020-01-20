from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# Automatically searches for approximate word equations like those in wvarith using combinations of words in a given list
# For variety, it avoids using multiple equations with the same pair

# Name of model file, output of wvgen.py
modelf = "mariomodel.bin"
# Name of file with list of words, separated by "," or newlines with no commas
wordf = "marionames.txt"
# Number of equations printed
num_outputs = 10


def approxLinear(model, words):
    outputs = []
    for i, first in enumerate(words):
        firstoutputs = []
        for j, second in enumerate(words[i+1:]):
            pairoutputs = []
            for third in words[(i+1)+(j+1):]:
                result = model.wv.most_similar_cosmul(
                    positive=[first, second], negative=[third], topn=2)
                top = result[0]
                if result[0][0] in [first, second, third]:
                    top = result[1]
                pairoutputs.append(
                    ([first, second, third], top[0], top[1]))
            result = model.wv.most_similar_cosmul(
                positive=[first, second], negative=[], topn=2)
            top = result[0]
            if result[0][0] in [first, second, third]:
                top = result[1]
            cont = False
            for o in outputs:
                if o[0][2] == third:
                    cont = True
                    # if o[2] <= top[1]:
                    # outputs.remove(o)
                    break
            if cont:
                break
            pairoutputs.append(
                ([first, second, "(None)"], top[0], top[1]))
            # Pick the top 2 best matches from this first/second pair
            pairoutputs.sort(key=lambda x: -x[2])
            pairoutputs = pairoutputs[:2]
            firstoutputs += pairoutputs
        firstoutputs.sort(key=lambda x: -x[2])
        firstoutputs = firstoutputs[:3]
        outputs += firstoutputs
        outputs.sort(key=lambda x: -x[2])
        outputs = outputs[:num_outputs]
    for o in outputs:
        print(o[0], o[1], o[2])


if __name__ == '__main__':
    model = Word2Vec.load("model/"+modelf)

    # Get words from file
    w = open("list/"+wordf, "r")
    words = ",".join(w.read().split("\n"))
    w.close()

    # Handle multi-word expressions
    words = [' '.join(word_tokenize(x)) for x in words.split(",")]
    # Remove words not in model
    words = [f for f in filter(lambda x: x in model.wv.vocab.keys(), words)]
    approxLinear(model, words)
