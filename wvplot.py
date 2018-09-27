from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Color of labels in each group (dark colors recommended, defaults to black)
# Hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["red", "turquoise", "orange", "forestgreen", "purple", "teal",
          "navy",  "slategrey", "olive", "maroon", "peru", "orangered", "crimson"]


def plot2D(model, wordgroups, default=True):
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in wordgroups):
            vocab[v] = model.wv.vocab[v]
    X = model[vocab]
    pca = PCA(n_components=2) if default else PCA(n_components=3)
    result = pca.fit_transform(X)
    words = list(vocab)
    pyplot.scatter(result[:, 0], result[:, 1]) if default else pyplot.scatter(
        result[:, 1], result[:, 2])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            pyplot.annotate(
                word, xy=(result[i, 0], result[i, 1]), color=colors[g] if g < len(colors) else "black", fontsize=8 if g < len(sizes) else 10) if default else pyplot.annotate(
                    word, xy=(result[i, 1], result[i, 2]), color=colors[g] if g < len(colors) else "black", fontsize=8 if g < len(sizes) else 10)
    pyplot.show()


def plot3D(model, wordgroups, default=True):
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in wordgroups):
            vocab[v] = model.wv.vocab[v]
    X = model[vocab]
    pca = PCA(n_components=3) if default else PCA(n_components=4)
    result = pca.fit_transform(X)
    words = list(vocab)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2]) if default else ax.scatter(
        result[:, 1], result[:, 2], result[:, 3])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            ax.text(result[i, 0], result[i, 1], result[i, 2], word, None, color=colors[g] if g < len(
                colors) else "black", fontsize=8 if g < len(sizes) else 10) if default else ax.text(result[i, 1], result[i, 2], result[i, 3], word, None, color=colors[g] if g < len(
                    colors) else "black", fontsize=8 if g < len(sizes) else 10)
    pyplot.show()


def plotVecs(type, model, wordgroups, default=True):
    f = open("list/" + wordgroups[0][0], "r", encoding="utf-8")
    groups = [g.split(",") for g in f.read().split("\n")]
    f.close()
    if "2" in type:
        plot2D(model, groups, default)
    else:
        plot3D(model, groups, default)


file = input("Model filename (end with .bin)\nmodel/")
type = input(
    "2D or 3D, main or alternate PCA axes (2D | 3D | 2D-alt | 3D-alt)\nType:")
words = input("File containing words to plot\nlist/")
# TODO: optional clustering to create groups automatically
plotVecs(type, Word2Vec.load("model/" + file), words, "alt" in type)
