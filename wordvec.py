from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

# Lost (TV) characters
# group1 = ["Jack", "Locke", "Kate", "Hurley", "Sawyer", "Sayid", "Jin",
#           "Sun", "Claire", "Charlie", "Michael", "Boone", ]  # Main cast
# group2 = ["Ben", "Juliet", "Richard", "Ethan", "Tom", "Mikhail"]  # Others
# group3 = ["Walt", "Rose", "Libby", "Eko", "Mr._Eko",
#           "Ana_Lucia", "Bernard", "Goodwin"]  # Secondary cast
# group4 = ["Desmond", "Jacob", "Daniel", "Charlotte", "Miles",
#           "Penny", "Ilana", "Pierre", "Frank"]  # Later season main cast
# group5 = ["Minkowski", "Roger", "Horace", "Radzinski"]  # Dharma
# group6 = ["Alex", "Danielle", "Arzt", "Vincent", "Keamy",
# "Widmore", "Nikki", "Paulo"]  # Tertiary cast

# Danganronpa franchise characters
# group1 = ["Yasuhiro",  "Mukuro", "Junko", "Celestia",  "Sakura", "Hifumi", "Mondo", "Chihiro",
#           "Kiyotaka",  "Byakuya", "Toko", "Leon", "Sayaka", "Makoto", "Kyoko", "Aoi", ]
# # "Hagakure", "Ikusaba", "Enoshima","Ludenberg", "Ogami", "Yamada", "Owada", "Fujisaki","Ishimaru","Togami",  "Fukawa",  "Kuwata", "Maizono",  "Naegi", "Kirigiri","Asahina"
# group2 = ["Peko",  "Akane", "Teruteru", "Nagito",  "Sonia", "Fuyuhiko",  "Kazuichi",
#           "Chiaki", "Mahiru",  "Hiyoko",  "Hajime", "Mikan",  "Ibuki", "Nekomaru",  "Gundham", ]
# # "Pekoyama", "Owari", "Hanamura", "Komaeda", "Nevermind","Kuzuryu","Soda", "Nanami", "Koizumi","Saionji", "Hinata","Tsumiki","Mioda", "Nidai","Tanaka"
# group3 = ["Korekiyo", "Angie",  "Kirumi", "Kokichi" "Tsumugi",  "Tenko", "Kaede",
#           "Kaito", "Maki", "Shuichi", "Ryoma",  "Rantaro", "Himiko",  "Gonta", "Miu", "K1-B0"]
# # "Shinguji", "Yonaga", "Tojo", , "Oma", "Shirogane", "Chabashira",  "Akamatsu", "Momota", "Harukawa",  "Saihara", "Hoshi", "Amami", "Yumeno", "Gokuhara",  "Iruma",
# group4 = ["Komaru", "Nagisa", "Shingetsu", "Jataro", "Kemuri",
#           "Kotoko", "Utsugi", "Monaca", "Masaru", "Daimon"]
# group5 = ["Monokuma", "Monomi", "Usami", "Kurokuma", "Shirokuma",
#           "Monokid", "Monophanie", "Monodam", "Monotaro", "Monosuke"]
# group6 = ["Keebo", "Imposter", "Kiyondo", "Ishida", "Kiyo", "Taka", "Hina",
#           "Hiro", "Mechamaru", "Minimaru", "Genocide", "Genocider", "Jack", "Jill", "Syo"]

# League of Legends characters by role
# group1 = ["Aatrox", "Camille", "Cho'Gath", "Darius", "Dr._Mundo", "Fiora", "Gangplank", "Garen", "Gnar", "Illaoi", "Irelia", "Jax", "Jayce", "Kayle", "Kled",
#           "Malphite", "Maokai", "Nasus", "Ornn", "Pantheon", "Poppy", "Renekton", "Riven", "Rumble", "Shen", "Singed", "Sion", "Teemo", "Tryndamere", "Wukong", "Yorick"]
# group2 = ["Amumu", "Elise", "Evelynn", "Fiddlesticks", "Gragas", "Graves", "Hecarim", "Ivern", "Jarvan IV", "Kayn", "Kha'Zix", "Kindred", "Lee_Sin", "Master_Yi", "Nautilus", "Nidalee",
#           "Nocture", "Nunu_&_Willump", "Olaf", "Rammus", "Rek'Sai", "Rengar", "Sejuani", "Shaco", "Shyvana", "Skarner", "Trundle", "Udyr", "Vi", "Volibear", "Warwick", "Xin_Zhao", "Zac"]
# group3 = ["Ahri", "Akali", "Anivia", "Annie", "Aurelion_Sol", "Azir", "Brand", "Cassiopeia", "Diana", "Ekko", "Fizz", "Galio", "Heimerdinger", "Karthus", "Kassadin", "Katarina", "Kennen", "LeBlanc",
#           "Lissandra", "Lux", "Malzahar", "Mordekaiser", "Orianna", "Ryze", "Swain", "Syndra", "Taliyah", "Talon", "Twisted_Fate", "Veigar", "Vel'Koz", "Viktor", "Vladimir", "Xerath", "Yasuo", "Zed", "Ziggs", "Zoe"]
# group4 = ["Alistar", "Bard", "Blitzcrank", "Braum", "Janna", "Karma", "Leona", "Lulu", "Morgana",
#           "Nami", "Pyke", "Rakan", "Sona", "Soraka", "Tahm_Kench", "Taric", "Thresh", "Zilean", "Zyra"]
# group5 = ["Ashe", "Caitlyn", "Corki", "Draven", "Ezreal", "Jhin", "Jinx", "Kai'Sa", "Kalista", "Kog'Maw",
#           "Lucian", "Miss_Fortune", "Quinn", "Sivir", "Tristana", "Twitch", "Urgot", "Varus", "Vayne", "Xayah"]
# group6 = ["Valor", "Urf"]

# The Office cast
group1 = ["Michael", "Jim", "Pam", "Dwight", "Oscar", "Angela", "Creed"
          "Kevin", "Ryan", "Kelly", "Stanley", "Toby", "Phyllis", "Meredith"]
group2 = ["Andy", "Karen", "Martin", "Tony", "Hannah", "Josh"]
group3 = ["Diangelo", "Charles", "Holly", "Pete", "Nellie",
          "Erin", "Robert", "Clark", "Gabe", "Cathy", "Jo"]
group4 = ["Darryl", "Roy", "Madge", "Hidetoshi", "Nate", "Val"]
group5 = ["David", "Jan", "Todd", "Hunter"]
# Groups of words to plot
groups = [group1, group2, group3, group4, group5]
# Color of labels in each group (dark colors recommended, defaults to black)
# Hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["red", "blue", "orange", "forestgreen", "teal", "purple"]
# Font size of labels in each group (default 10)
sizes = [10, 10, 10, 8, 8, 6]


def makePlot(model, default=True):
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in groups):
            vocab[v] = model.wv.vocab[v]
    X = model[vocab]
    pca = PCA(n_components=2) if default else PCA(n_components=3)
    result = pca.fit_transform(X)
    words = list(vocab)
    pyplot.scatter(result[:, 0], result[:, 1]) if default else pyplot.scatter(
        result[:, 1], result[:, 2])
    for g, group in enumerate(groups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            pyplot.annotate(
                word, xy=(result[i, 0], result[i, 1]), color=colors[g] if g < len(colors) else "black", fontsize=sizes[g] if g < len(sizes) else 10) if default else pyplot.annotate(
                    word, xy=(result[i, 1], result[i, 2]), color=colors[g] if g < len(colors) else "black", fontsize=sizes[g] if g < len(sizes) else 10)
    pyplot.show()


def plot3D(model, default=True):
    vocab = {}
    for v in model.wv.vocab.keys():
        if any(v in g for g in groups):
            vocab[v] = model.wv.vocab[v]
    X = model[vocab]
    pca = PCA(n_components=3) if default else PCA(n_components=4)
    result = pca.fit_transform(X)
    words = list(vocab)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2]) if default else ax.scatter(
        result[:, 1], result[:, 2], result[:, 3])
    for g, group in enumerate(groups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            ax.text(result[i, 0], result[i, 1], result[i, 2], word, None, color=colors[g] if g < len(
                colors) else "black", fontsize=sizes[g] if g < len(sizes) else 10) if default else ax.text(result[i, 1], result[i, 2], result[i, 3], word, None, color=colors[g] if g < len(
                    colors) else "black", fontsize=sizes[g] if g < len(sizes) else 10)
    pyplot.show()


def makeModel(files, outputf, mode):
    sentences = []
    for file in files:
        f = open("text/" + file, "r", encoding="utf-8")
        sentences += [word_tokenize(s) for s in sent_tokenize(f.read())]
    if "big" in mode:
        bigram = Phrases(sentences, max_vocab_size=40000)
        bigram_phraser = Phraser(bigram)
        model2 = Word2Vec(bigram_phraser[sentences], min_count=1)
        model2.save("models/" + outputf)
    else:
        model = Word2Vec(sentences, min_count=1)
        model.save("models/" + outputf)


def mostSim(model):
    pos = input("Enter positives separated by ' '\n").split(" ")
    if len(pos) == 0:
        print("Error: empty positive\n")
        mostSim(model)
    neg = input("Enter negatives separated by ' '\n").split(" ")
    try:
        print([x[0] for x in model.most_similar_cosmul(positive=pos, negative=neg)])
    except KeyError:
        print("Error: unknown word\n")
    mostSim(model)


def approxLinear(model):
    articles = ["the", "a", "an"]
    pronouns = ["all", "another", "any", "anybody", "anyone", "anything", "as", "both", "each", "either", "everybody", "everyone", "everything", "few", "he", "her", "hers", "herself", "him", "himself", "his", "I", "it", "itself", "many", "me", "mine", "most", "my", "myself", "neither", "no one", "nobody", "none", "nothing", "one", "other", "others", "our",
                "ours", "ourselves", "several", "she", "some", "somebody", "someone", "something", "such", "that", "thee", "their", "theirs", "them", "themselves", "these", "they", "thine", "this", "those", "thou", "thy", "us", "we", "what", "whatever", "which", "whichever", "who", "whoever", "whom", "whomever", "whose", "you", "your", "yours", "yourself", "yourselves"]
    prepositions = ["aboard", "about", "above", "absent", "across", "after", "against", "along", "alongside", "amid", "amidst", "among", "amongst", "around", "as", "aside", "astride", "at", "athwart", "atop",                    "barring", "before", "behind", "below", "beneath", "beside", "besides", "between", "betwixt", "beyond", "but", "by", "circa", "concerning", "despite", "down", "during", "except", "excluding",
                    "failing", "following", "for", "from", "given", "in", "including", "inside", "into", "like", "mid", "minus", "near", "next", "notwithstanding", "of", "off", "on", "onto", "opposite", "out", "outside", "over",                    "pace", "past", "per", "plus", "pro", "qua", "regarding", "round", "save", "since", "than", "through_", "thru_", "throughout", "thruout", "till", "time", "to", "toward", "towards", "under", "underneath", "unlike", "until", "up", "upon", "versus", "vs.", "via", "vice", "with", "within", "without", "worth",
                    "according_to", "ahead_of", "as_of", "as_per", "as_regards", "aside_from", "because_of", "close_to", "due_to", "except_for", "far_from", "in_to", "into", "inside_of", "instead_of", "near_to", "next_to", "on_to", "onto", "out_from", "out_of", "outside_of", "owing_to", "prior_to", "pursuant_to", "regardless_of", "subsequent_to", "thanks_to", "that_of", "as_far_as", "as_well_as", "by_means_of", "in_accordance_with", "in_addition_to", "in_case_of", "in_front_of", "in_lieu_of", "in_place_of", "in_point_of", "in_spite_of", "on_account_of", "on_behalf_of", "on_top_of", "with_regard_to", "with_respect_to"]
    # Remove words with less meaning
    wordsToIgnore = articles + pronouns  # + prepositions
    for word in wordsToIgnore:
        model.wv.vocab.pop(word, None)
    # Ignore similarity to self
    checkedPairs = [(w, w) for w in model.wv.vocab.keys()]
    for first in model.wv.vocab.keys():
        for second in model.wv.vocab.keys():
            if (second, first) in checkedPairs:
                continue
            for third in model.wv.vocab.keys():
                for result in model.most_similar_cosmul(positive=[first, second], negative=[third], topn=3):
                    if result[1] > 10:
                        print(first + "\t+\t" + second + "\t-\t" +
                              third + "\t=\t" + result[0] + "\tsimilarity=" + str(result[1]))
                    else:
                        break
            checkedPairs.append((first, second))


def main():
    # Get all user input
    mode = input(
        "Enter mode: gen-uni | gen-big | cmp | plot | plot-alt | plot3d | plot3d-alt | sim | help \n")
    if "gen" in mode:
        file = input(
            "Enter .txt filename(s), comma separated e.g. file1.txt,file2.txt,file3.txt)\n")
        outputf = input("Enter output filename (ending with .bin)\n")
        makeModel(file.split(","), outputf, mode)
    elif mode == "cmp":
        file = input("Enter filename (should end with .bin)\n")
        model = Word2Vec.load("models/" + file)
        # model = KeyedVectors.load_word2vec_format(argv[1], binary=False)
        mostSim(model)
    elif "plot" in mode:
        file = input("Enter filename (should end with .bin)\n")
        if mode == "plot":
            makePlot(Word2Vec.load("models/" + file))
        if mode == "plot-alt":
            makePlot(Word2Vec.load("models/" + file), False)
        elif mode == "plot3d":
            plot3D(Word2Vec.load("models/" + file))
        elif mode == "plot3d-alt":
            plot3D(Word2Vec.load("models/" + file), False)
    elif mode == "sim":
        file = input("Enter filename (should end with .bin)\n")
        model = Word2Vec.load("models/" + file)
        # model = KeyedVectors.load_word2vec_format(argv[1], binary=False)
        approxLinear(model)
    else:
        print("Modes: \n gen-uni: generate with unigrams\ngen-big: generate with bigrams\nplot1: plot along top two PCA axes\nplot2: plot along second and third PCA axes\ncmp: custom arithmetic with word vectors\nsim: find strongest A+B-C=D similarities\n")
        main()


main()
