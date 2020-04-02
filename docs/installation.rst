
.. _instalation:

Installation
============

Using pip::

    pip install aikit

In order to use the full functionnalities of aikit you can also install additionnal packages :

 - graphviz : to have a nice representation of the graph of models
 - lightgbm : to use lightgbm in the auto-ml
 - nltk and gensim : to have advanced text encoder
 - nltk corpus to clean text
 
To install everything you can do the following::

    pip install lightgbm
    pip install gensim
    pip install nltk
    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords
    conda install graphviz
