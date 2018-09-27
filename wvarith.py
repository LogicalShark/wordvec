from gensim.models import Word2Vec


def mostSim(model):
    pos = input("Enter positives separated by ','\n").split(",")
    if len(pos) == 0:
        print("Error: empty positive\n")
        mostSim(model)
    neg = input("Enter negatives separated by ','\n").split(",")
    try:
        print([x[0] for x in model.most_similar_cosmul(positive=pos, negative=neg)])
    except KeyError:
        print("Error: word not in vocabulary or incorrect delimiter between words\n")
    mostSim(model)


file = input("Model filename (should end with .bin)\nmodel/")
model = Word2Vec.load("model/" + file)
# TODO: KeyedVectors?
# model = KeyedVectors.load_word2vec_format(argv[1], binary=False)
mostSim(model)
