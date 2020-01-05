from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec

# Plots words in a given list based on PCA axes

# Path to model file, output of wvgen.py
modelf = "model/hkmodel.bin"
# Path to file with list words to be plotted, words separated by ","
# and optionally, groups separated by newlines
wordf = "list/hknames.txt"
# PCA axes to plot on, the most relevant are [0,1] or [1,2] for 2D or [0,1,2] or [1,2,3] for 3D
axes = [0, 1, 2]


# To differentiate groups in the graph, you can give the labels a corresponding color or font size
# e.g. words in the first group will be red, words in the second group will be turquoise, etc.

# Color of words in each group (defaults to black if too many groups)
# I recommend dark colors, use hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:pink", "tab:olive", "tab:pink",
          "tab:cyan", "tab:gray", "forestgreen", "teal", "navy", "maroon", "peru", "orangered", "crimson"]
# Font sizes of words in each group (defaults to 10)
sizes = [] # [10, 10, 10, 10, 10, 10, 8, 8, 8, 8]


def plot2D(result, wordgroups):
    words = list(vocab)
    pyplot.scatter(result[:, axes[0]], result[:, axes[1]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            pyplot.annotate(word, xy=(result[i, axes[0]], result[i, axes[1]]), color=colors[g] if g < len(
                colors) else "black", fontsize=sizes[g] if g < len(sizes) else 10)
    pyplot.show()


def plot3D(result, wordgroups):
    words = list(vocab)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, axes[0]], result[:, axes[1]], result[:, axes[2]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            ax.text(result[i, axes[0]], result[i, axes[1]], result[i, axes[2]], word, None, color=colors[g] if g < len(
                colors) else "black", fontsize=sizes[g] if g < len(sizes) else 10)
    pyplot.show()


if __name__ == '__main__':
    # TODO: clustering to create groups automatically
    f = open(wordf, "r", encoding="utf-8")
    groups = [g.split(",") for g in f.read().split("\n")]
    f.close()
    model = Word2Vec.load(modelf)
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in groups):
            vocab[v] = model.wv.vocab[v]
    coords = model.wv[vocab]
    pca = PCA(n_components=max(axes)+1)
    result = pca.fit_transform(coords)
    if len(axes) > 2:
        plot3D(result, groups)
    else:
        plot2D(result, groups)
