from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.scripts.glove2word2vec import glove2word2vec


def makeModel(files, outputf, mode):
    sentences = []
    for file in files:
        f = open("text/" + file, "r", encoding="utf-8")
        sentences += [word_tokenize(s) for s in sent_tokenize(f.read())]
        f.close()
    if "big" in mode:
        bigram = Phrases(sentences, max_vocab_size=40000)
        bigram_phraser = Phraser(bigram)
        model2 = Word2Vec(bigram_phraser[sentences], min_count=1)
        model2.save("model/" + outputf)
    else:
        model = Word2Vec(sentences, min_count=1)
        model.save("model/" + outputf)


type = input(
    "Use unigrams (faster, smaller) or bigrams (uni | big)\nType:")
file = input(
    "Input filename(s) (onlyfile.txt | firstfile.txt,secondfile.txt...)\ntext/")
outputf = input(
    "Output filename, end with .bin\nmodel/")
makeModel(files, outputf, type)
