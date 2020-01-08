from gensim.models import Word2Vec

# For doing arithmetic with word vectors
# e.g. king + woman - man ~= queen, Paris + Italy - France ~= Rome

# Name of model file, output of wvgen.py
modelf = "hkmodel.bin"
# Words to add
positives = ["City of Tears", "Howling Cliffs"]
# Words to subtract
negatives = ["Greenpath"]
# Number of results
n_results = 3;

def mostSim(model):
    try:
        print(model.wv.most_similar_cosmul(positive=[' '.join(
            x) for x in positives], negative=[' '.join(x) for x in negatives], topn=n_results))
    except KeyError:
        print("Error: input word not in vocabulary\n")

if __name__ == "__main__":
    model = Word2Vec.load("model/"+modelf)
    mostSim(model)
