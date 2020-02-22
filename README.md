# Word Vectors with Python

## Setup
Download everything, or just the Python files (but use `text/`, `list/`, and `model/` directories)

Install packages -- `conda install gensim nltk scikit-learn matplotlib`
On Windows 10 I strongly recommend using an Anaconda environment and installing a C compiler like MinGW to speed up wvgen.py

The python files, look at the parameters at the top and customize based on your own filenames and desired outputs

`wvgen.py` generates the model, a prerequisite for all other steps. Input files go in `text/` and the output is saved in `model/`

`wvplot.py` plots words from a file in `list/`

`wvarith.py` evaluates manual word vector comparisons, manual input

`wvlinear.py` finds approximate word vector equations from a file in `list/`

## Useful input text sources
Database dumps from Fandom wikis (something.wikia.com/wiki/Special:Statistics)

[Twitter API](https://developer.twitter.com/en/docs)

[Public domain books](http://gutenberg.org)

[Classic texts](http://classics.mit.edu)

[Misc. Corpora](https://en.wikipedia.org/wiki/List_of_text_corpora)

## Included Examples:
`model/` includes models for Super Mario, League of Legends, The Office (TV), Star Wars, Danganronpa, and Hollow Knight

`list/` includes some names from these examples to be plotted

`text/` includes xml dumps from the smaller wikis