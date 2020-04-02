.. aikit documentation master file, created by
   sphinx-quickstart on Fri Jun 15 13:49:16 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AIkit's documentation!
=================================


aikit stands for Artificial Intelligent tool Kit and provides method to facilitate and accelerate the DataScientist job.

The optic is to provide tools to ease the repetitive part of the DataScientist job and so that he/she can focus on modelization.
This package is still in alpha and more features will be added.

This library is intended for the user who knows machine learning, knows python its data-science environnement (sklearn, numpy, pandas, ...) but doesn't want to spend too much time thinking about python technicallities in order to focus more on modelling.
The idea is to automatize or at least accelerate parts of the DataScientist job so that he/she can focus on what he/she does best. 
The more time spend on coding the less time spent on asking the rights questions and solving the problems.
It will also help to really use model in production and not just play with them.

The library is usefull if you ever asked yourself that type of questions :
 * How do I handle different type of data ?
 * I don't remember how to concatenate sparse array and dataframe ?
 * How can I retrieve the name of my features now that everything is a numpy array ?
 * I'd like to use sklearn but my data is in a DataFrame with strings object and I don't want to use 2 transformers just to encode the categorical features ?
 * How do I deal with Data with several types like text, number and categorical data ?
 * How can I quickly test models to see what work and what doesn't ?
 * ...
 
Here a quick summary of what is provided:
 * additional sklearn-like transformers to facilitate operations (categories encoding, missing value handling, text encoding, ...) : :ref:`transformer`
 * an extension of sklearn Pipeline that handle generic composition of transformations : :ref:`graph_pipeline`
 * a framework to automatically test machine learning models : :ref:`auto_ml`
 * helper functions to accelerate the *day-to-day*
 * ...


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation

   :titlesonly:
   Getting Started <../notebooks/GettingStarted.ipynb>
    
   examples
   graph_pipeline
   auto_ml

   transformer
   
   other_functionnalities
   adding_new_models
   contribution


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
