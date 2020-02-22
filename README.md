# Word Vectors with Python

## Setup
Download everything, or just download the Python files (they assume you'll have the same directory structure though)

Install packages -- `conda install gensim nltk scikit-learn matplotlib`
If on Windows 10, I strongly recommend installing a C compiler like MinGW and using an Anaconda environment for speed with wvgen.py

The python files, look at the parameters at the top and customize based on your own filenames and desired outputs

`wvgen.py` generates the model (necessary for all other steps, put input files in `text/`)

`wvplot.py` plots words from a file in `list/`

`wvlinear.py` finds approximate word vector equations

`wvarith.py` evaluates manual word vector comparisons

## Useful data sources:
Database dumps from Fandom wikis (something.wikia.com/wiki/Special:Statistics)

[Public domain books](http://gutenberg.org)

[Twitter API](https://developer.twitter.com/en/docs)

[Misc. Corpora](https://en.wikipedia.org/wiki/List_of_text_corpora)

[Classic texts](http://classics.mit.edu)

## Included Examples:
`model/` includes models for Super Mario, League of Legends, The Office (TV), Star Wars, Danganronpa, and Hollow Knight

`list/` includes some names from these examples to plot

`text/` includes xml dumps from the smaller wikis