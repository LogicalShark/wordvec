from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
# For doing arithmetic with word vectors
# e.g. king + woman - man ~= queen, Paris + Italy - France ~= Rome

# Name of model file, output of wvgen.py
modelf = "mariomodel.bin"

# Words to add and subtract
# Negative can be empty
# If both empty, prompts for input from command line
positives = []  # Mario, Princess
negatives = []  # Plumber
# Ex: Mario + Princess - Plumber = Peach

# Number of best fitting results to print
n_out = 5


def equation_result(pos, neg, model):
    try:
        # Handle multi-word expressions, assuming MWETokenizer separator=' ' in wvgen.py
        p = [' '.join(word_tokenize(x)) for x in pos]
        n = [' '.join(word_tokenize(x)) for x in neg]
        result = model.wv.most_similar_cosmul(
            positive=p, negative=n, topn=n_out)
        print([(x[0], round(x[1], 3)) for x in result])
    except KeyError:
        print("KeyError: input word not in vocabulary\n"+positives+"\n"+negatives)


if __name__ == "__main__":
    model = Word2Vec.load("model/"+modelf)

    # Get user input if needed
    if len(positives) == 0 and len(negatives) == 0:
        p = input("Enter positives, separated by \",\":").split(",")
        positives = list(filter(lambda x: len(x) > 0,
                                [w.strip() for w in p]))
        n = input("Enter negatives, separated by \",\":").split(",")
        negatives = list(filter(lambda x: len(x) > 0,
                                [w.strip() for w in n]))

    equation_result(positives, negatives, model)
