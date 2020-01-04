from gensim.models import Word2Vec

# Automatically searches for approximate word equations like those in wvarith using combinations of words in a given list
# For variety, it avoids using too many of the same pairs

# Path to model file, output of wvgen.py
modelf = "model/model.bin"
# Path to file with list of words, separated by "," or newlines with no commas
wordf = "list/champions.txt"

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
            pairoutputs = []
            for third in words:
                try:
                    for result in model.wv.most_similar_cosmul(positive=[first, second], negative=[third], topn=1):
                        pairoutputs.append(
                            (first, second, third, result[0], result[1]))
                except:
                    pass
            pairoutputs.sort(key=lambda x: x[4])
            pairoutputs = pairoutputs[:1]
            outputs += pairoutputs
        outputs.sort(key=lambda x: x[4])
        outputs = outputs[:10]
    for (first, second, third, out, sim) in outputs:
        print(first + "\t+\t" + second + "\t-\t" +
              third + "\t=\t" + out + "\nsimilarity=" + str(sim))

if __name__ == '__main__':
    model = Word2Vec.load(modelf)
    w = open(wordf, "r")
    words = ",".join(w.read().split("\n"))
    w.close()
    approxLinear(model, words.split(","))
