from gensim.models import Word2Vec

# For doing arithmetic with word vectors
# e.g. king + woman - man ~= queen, Paris + Italy - France ~= Rome

# Path to model file, output of wvgen.py
modelf = "model/model.bin"
# Words to add
positives = ["Alesia", "Germania"]
# Words to subtract
negatives = ["Gallia"]
# Number of results
n_results = 3;

def mostSim(model):
    try:
        print([x[:n_results] for x in model.wv.most_similar_cosmul(positive=positives, negative=negatives)])
    except KeyError:
        print("Error: word not in vocabulary\n")

if __name__ == "__main__":
    model = Word2Vec.load(modelf)
    mostSim(model)
