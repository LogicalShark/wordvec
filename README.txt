Word Vectors with Python

SETUP
Install packages:
pip install gensim nltk scikit-learn matplotlib
Add one or more file to the text directory
~200MB is fine, ~1GB file froze my computer so watch out

In any of the below files, look at the comments at the top and customize based on your own files
wvgen.py generates the model (necessary for all other steps)
wvplot.py plots words

Compare word vectors with wvarith.py

Find approximate equations with wvlinear.py

Useful data sources:
wikia.com will often have database dumps on https://[name].wikia.com/wiki/Special:Statistics
gutenberg.org has free public domain books in plaintext
classics.mit.edu has classic texts

Included Examples:
model/ includes models for Danganronpa, Hollow Knight, League of Legends, and The Office (TV)
text/ includes xml dumps from the DR and HK wikis