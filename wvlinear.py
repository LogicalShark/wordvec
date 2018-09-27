from gensim.models import Word2Vec


def approxLinear(model, words):
    if len(words) == 1:
        f = open("list/" + words[0], "r", encoding="utf-8")
        words = []
        for group in f.read().split("\n"):
            words += group.split(",")
            if "" in words:
                words.remove("")
        f.close()
    outputs = []
    for i, first in enumerate(words):
        for j, second in enumerate(words[i:]):
            for third in words:
                for result in model.most_similar_cosmul(positive=[first, second], negative=[third], topn=1):
                    outputs.append(
                        (first, second, third, result[0], result[1]))
        outputs.sort(key=lambda x: x[4])
        outputs = outputs[:10]
    for (first, second, third, out, sim) in outputs:
        print(first + "\t+\t" + second + "\t-\t" +
              third + "\t=\t" + out + "\t\tsimilarity=" + str(sim))


file = input("Model filename (should end with .bin)\nmodel/")
model = Word2Vec.load("model/" + file)
words = input(
    "Filename to use for input\nlist/")
approxLinear(model, words.split(","))
