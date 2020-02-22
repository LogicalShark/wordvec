from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
# Automatically searches for approximate word equations like those in wvarith using combinations of words in a given list
# For variety, it avoids using multiple equations with the same pair

# Name of model file, output of wvgen.py
modelf = "mariomodel.bin"
# Name of file with list of words, separated by "," or newlines with no commas
wordf = "mariochars.txt"
# Number of equations printed
num_outputs = 15


def filter_results(outputs):
    newoutputs = []
    # Sort by result similarity
    outputs.sort(key=lambda x: -x[1])
    # Track occurrences of each word
    occs = {w: 0 for w in [o[0][n] for n in range(4) for o in outputs]}
    # Don't delete more than necessary
    toDelete = len(outputs) - num_outputs
    # Filter to prevent too many repeats of the same word
    for o in outputs:
        # Increment occurrences
        for n in range(4):
            occs[o[0][n]] += 1
        # Check number of occurrences of each word
        uniqueResult = all([occs[o[0][n]] <= 4 for n in range(4)])
        # Don't delete if it's unique enough
        if uniqueResult or toDelete == 0:
            newoutputs.append(o)
        else:
            toDelete -= 1
    return newoutputs


def approx_linear(model, words):
    outputs = []
    # Iterate through all equations, don't use the same word twice in the equation
    for i, first in enumerate(words):
        for j, second in enumerate(words[i+1:]):
            for third in words[(i+1)+(j+1):]:
                # Find equation results
                result = model.wv.most_similar_cosmul(
                    positive=[first, second], negative=[third], topn=3)
                outputs += ([([first, second, third, result[n][0]],
                              result[n][1]) for n in range(3)])
        # Filter lowest similarity repeats
        outputs = filter_results(outputs)
    # Delete least similar
    outputs = outputs[:num_outputs]
    for o in outputs:
        print(o[0][0], "+", o[0][1], "-", o[0][2],
              "=", o[0][3]+" :", round(o[1], 3))


if __name__ == '__main__':
    model = Word2Vec.load("model/"+modelf)

    # Get words from file
    w = open("list/"+wordf, "r")
    words = ",".join(w.read().split("\n"))
    w.close()

    # Handle multi-word expressions, assuming MWETokenizer separator=' ' in wvgen.py
    words = [' '.join(word_tokenize(x)) for x in words.split(",")]

    # Remove words not in vocabulary
    words = [f for f in filter(lambda x: x in model.wv.vocab.keys(), words)]

    approx_linear(model, words)
