from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

# Plots words in a given list based on PCA axes

# Model filename, output of wvgen.py
modelf = "leaguemodel.bin"
# Name of file with words to be plotted, separated by ","
# and groups separated by newlines
wordf = "lolchamps.txt"
# PCA axes to plot on, the most relevant are [0,1] or [1,2] for 2D or [0,1,2] or [1,2,3] for 3D
axes = [0, 1, 2]
# Number of groups to cluster into, use 0 to instead group based on lines in wordf
clusterK = 3


# To differentiate groups in the graph, you can give the labels a corresponding color or font size
# e.g. words in the first group will be red, words in the second group will be turquoise, etc.

# Color of words in each group (defaults to black)
# Dark colors are good for matplotlib's white background, use hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:olive", "tab:pink",
          "tab:cyan", "tab:gray", "forestgreen", "teal", "navy", "maroon", "peru", "orangered", "crimson"]
# Font sizes of words in each group (defaults to 10)
sizes = []


def plot2D(result, wordgroups):
    pyplot.scatter(result[:, axes[0]], result[:, axes[1]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            coord = (result[i, axes[0]], result[i, axes[1]])
            color = colors[g] if g < len(colors) else "black"
            size = sizes[g] if g < len(sizes) else 10
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
            color = colors[g] if g < len(colors) else "black"
            size = sizes[g] if g < len(sizes) else 10
            ax.text(result[i, axes[0]], result[i, axes[1]],
                    result[i, axes[2]], word, color=color, fontsize=size)


if __name__ == '__main__':
    model = Word2Vec.load("model/" + modelf)

    # Extract words to plot from file
    groups = []
    words = []
    for line in open("list/" + wordf, "r", encoding="utf-8").read().split("\n"):
        l = [' '.join(word_tokenize(x)) for x in line.split(",")]
        l = filter(lambda x: x in model.wv.vocab.keys(), l)
        groups.append(l)
        words += l

    # Get word vectors from model
    vocab = {w: model.wv.vocab[w] for w in words}
    coords = model.wv[vocab]

    # Create axes to plot on
    pca = PCA(n_components=max(axes)+1)
    result = pca.fit_transform(coords)
    
    # Assign groups based on clustering
    if clusterK > 0:
        estimator = KMeans(init='k-means++', n_clusters=clusterK, n_init=10)
        estimator.fit_predict(model.wv[vocab])
        groups = [[] for n in range(clusterK)]
        for i, w in enumerate(vocab.keys()):
            group = estimator.labels_[i]
            groups[group].append(w)
    if len(axes) > 2:
        plot3D(result, groups)
    else:
        plot2D(result, groups)
    pyplot.show()
