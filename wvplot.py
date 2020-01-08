from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# Plots words in a given list based on PCA axes

# Model filename, output of wvgen.py
modelf = "hkmodel.bin"
# Name of file with words to be plotted, separated by ","
# and groups separated by newlines
wordf = "hknames.txt"
# PCA axes to plot on, the most relevant are [0,1] or [1,2] for 2D or [0,1,2] or [1,2,3] for 3D
axes = [0, 1, 2]
# Cluster automatically instead of using the newline separation in wordf for groups
cluster = True
# Groups if using automatic clustering (k-means++)
k = 5


# To differentiate groups in the graph, you can give the labels a corresponding color or font size
# e.g. words in the first group will be red, words in the second group will be turquoise, etc.

# Color of words in each group (defaults to black if too many groups)
# I recommend dark colors, use hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:pink", "tab:olive", "tab:pink",
          "tab:cyan", "tab:gray", "forestgreen", "teal", "navy", "maroon", "peru", "orangered", "crimson"]
# Font sizes of words in each group (defaults to 10)
sizes = [12, 12, 10, 8, 8, 8]


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
    groups = []
    words = []
    for line in open("list/" + wordf, "r", encoding="utf-8").read().split("\n"):
        groups.append(line.split(","))
        words += line.split(",")
    model = Word2Vec.load("model/" + modelf)
    vocab = {}
    for w in words:
      if w in model.wv.vocab.keys():
            vocab[w] = model.wv.vocab[w]
    coords = model.wv[vocab]
    pca = PCA(n_components=max(axes)+1)
    result = pca.fit_transform(coords)

    if cluster:
      estimator = KMeans(init='k-means++', n_clusters=k, n_init=10)
      estimator.fit_predict(model.wv[vocab])
      groups = [[] for n in range(k)]
      for i,w in enumerate(vocab.keys()):
          group = estimator.labels_[i]
          groups[group].append(w)
    if len(axes) > 2:
        plot3D(result, groups)
    else:
        plot2D(result, groups)
