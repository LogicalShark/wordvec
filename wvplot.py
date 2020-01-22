from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

# Plots words in a given list based on PCA axes

# Model filename, output of wvgen.py
modelf = "hkmodel.bin"
# Name of file with words to be plotted, separated by ","
# and groups separated by newlines
wordf = "hkchars.txt"
# PCA axes to plot on, the most relevant are [0,1] or [1,2] for 2D or [0,1,2] or [1,2,3] for 3D
axes = [0, 1]
# Number of groups to cluster into, if left at 0 grouping is based on line separation in wordf
clusterK = 3


# To differentiate groups in the graph, you can give the labels a corresponding color or font size
# e.g. words in the first group will be red, words in the second group will be blue, etc.

# Color of words in each group, uses default if too many groups
# Dark colors are good for matplotlib's white background, use hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["tab:red", "tab:blue", "tab:green", "tab:orange",
          "tab:purple", "tab:olive", "tab:pink", "tab:cyan", "tab:gray"]
defaultcolor = "black"

# Font sizes of words in each group
sizes = []
defaultsize = 16


def plot2D(result, wordgroups):
    pyplot.scatter(result[:, axes[0]], result[:, axes[1]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            coord = (result[i, axes[0]], result[i, axes[1]])
            color = colors[g] if g < len(colors) else defaultcolor
            size = sizes[g] if g < len(sizes) else defaultsize
            pyplot.annotate(word, xy=coord, color=color, fontsize=size)


def plot3D(result, wordgroups):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, axes[0]], result[:, axes[1]], result[:, axes[2]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            color = colors[g] if g < len(colors) else defaultcolor
            size = sizes[g] if g < len(sizes) else defaultsize
            ax.text(result[i, axes[0]], result[i, axes[1]],
                    result[i, axes[2]], word, color=color, fontsize=size)


def get_groups(wordf, model):
    # Extract words to plot from file
    groups = []
    words = []
    for line in open("list/" + wordf, "r", encoding="utf-8").read().split("\n"):
        l = [' '.join(word_tokenize(x)) for x in line.split(",")]
        l = filter(lambda x: x in model.wv.vocab.keys(), l)
        groups.append(l)
        words += l

    # Get word vectors from model
    vecs = {w: model.wv.vocab[w] for w in words}

    # Assign groups if using clustering
    if clusterK > 0:
        estimator = KMeans(init='k-means++', n_clusters=clusterK, n_init=10)
        estimator.fit_predict(model.wv[vecs])
        groups = [[] for n in range(clusterK)]
        for i, w in enumerate(vecs.keys()):
            group = estimator.labels_[i]
            groups[group].append(w)

    return words, groups, vecs


if __name__ == '__main__':
    model = Word2Vec.load("model/" + modelf)

    # Get groups from file or by clustering
    words, groups, vecs = get_groups(wordf, model)

    coords = model.wv[vecs]

    # Create axes to plot on
    pca = PCA(n_components=max(axes)+1)
    result = pca.fit_transform(coords)

    if len(axes) > 2:
        plot3D(result, groups)
    else:
        plot2D(result, groups)
    pyplot.show()
