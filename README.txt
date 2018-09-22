Word Vectors in Python

SETUP

Install packages:
  > pip install gensim nltk sklearn matplotlib

Add files to create word vectors from:
  Add a file to the directory text/ (~200MB txt files are fine, ~1GB file froze my computer)

GENERATE WORD VECTOR MODEL (required to do other functions)
  > python wvgen.py
  Model will be generated in directory model/
  
PLOT WORDS FROM WORD VECTOR MODEL
  > python wvplot.py
  
CUSTOM WORD VECTOR ARITHMETIC
  > python wvarith.py
  
WORD VECTOR LINEAR EQUATIONS
  > python wvlinear.py
