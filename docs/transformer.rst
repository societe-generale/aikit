
.. _transformer:

Transformer
===========

aikit offers some transformers to help process the data. Here is a brief description of them.

Some of those transformers are just relatively thin wrapper around what exists in sklearn, some are existing techniques packaged as transformers and some things built from scratch.

aikit transformers are built using a Model Wrapper

There is a more detailed explanation about the :class:`aikit.transformers.model_wrapper.ModelWrapper` class.
It will explain what the wrapper is doing and how to wrap new models. It will also explain some common functionnalities of all the transformers in aikit.


Wrapper Goal
------------
The aim of the wrapper is to provide a generic class to handle most of the redondant operations that we might want to apply in a transformer.
In particular it aims at making regular 'sklearn like' model more generic and more 'user friendly'.

Here are a few things the wrapper offer to aikit transformers :

 * automatic conversion of input/output into a given format (which is useful when chaining models and some on them accepts DataFrame, some don't, ...)
 * verification of type, shape of new data
 * shape conversion for model that only accept '1-dimensional' input 
 * automatic splits and concatenation of result for models that only work one column at a time (See : :ref:`CountVectorizerWrapper`)
 * generation of features_names and usage of those names when the output is a DataFrame
 * delay the creation of underlying model until the :func:`fit` is called. This allow to customize hyper-parameters based on the data (Ex : ``n_components`` can be a percentage of the number of columns given in input).
 * automatic or manual selection the columns the transformers is supposed to work on.
 
Let's take sklearn :class:`sklearn.feature_extraction.text.CountVectorizer` as an example.
The transformer has the logic implemented however it can sometimes be a little difficult to use :

 * if your data has more than one text column you need more than once CountVectorizer and you need to concatened the result
Indeed CountVectorizer work only on 1 dimensional input (corresponding to a text Serie or a text list)

 * if your data is relatively small you might want to retrieve a regular pandas DataFrame and not a scipy.sparse matrix which might not work with your following steps
 * you might want to have feature_names that not only correspond to the 'word/char' but also tells from which column it comes from. Example of such column name : 'text1_BAG_dog'
 
 * you might want to tell the CountVectorizer to work on specific columns (so that you don't have to take care of manually splitting your data)

As a consequence it also make the creation of a "sklearn compliant" model (ie : a model that works well within the sklearn infrastructure easy : clone, set_params, hyper-parameters search, ...)

Wrapping the model makes the creation of complexe pipleline like the in :ref:`graph_pipeline` a lot easier.

To sum up the aim of the wrapper is to separate :
 1. the logic of the transformer
 2. the *mechanical* data transformation, checks, ... needed to make the transformer robust and easy to use


Selection of the columns
------------------------
The transformers present in aikit are able to select the columns they work on via an hyper-parameter called 'columns_to_use'.

For example:
    
    from aikit.transformers import CountVectorizerWrapper
    vectorizer = CountVectorizerWrapper(columns_to_use=["text1","text2"])
    
the preceding vectorizer will encode "text1" and "text2" using bag-of-word.

The parameter 'columns_to_use' can be of several type :
 * list of strings  : list of columns by name (assuming a DataFrame input)
 * list of integers : list of columns by position (either a numpy array or a DataFrame
 * special string "all" : means all the columns are used
 * DataTypes.CAT  : use aikit detection of columns type to keep only *categorical* columns
 * DataTypes.TEXT : use aikit detection of columns type to keep only *textual* columns
 * DataTypes.NUM  : use aikit detection of columns type to keep only *numerical* columns
 * other string like 'object' : use pandas.select_dtype to filter based on type of column
 
    
Remark : when a list of string is used the 'use_regex' attribute can be set to true. In that case the 'columns_to_use' are regexes and the columns retrieved are the one that match one of the regexes.

Here are a few examples:

Encoding only one or two columns by name::

    from aikit.transformers import CountVectorizerWrapper
    vectorizer = CountVectorizerWrapper(columns_to_use=["text1","text2"])


Encoding only 'TEXT' columns::

    from aikit.transformers import CountVectorizerWrapper
    from aikit.enums import DataTypes
    vectorizer = CountVectorizerWrapper(columns_to_use=DataTypes.TEXT)

Encoding only 'object' columns::

    from aikit.transformers import CountVectorizerWrapper
    from aikit.enums import DataTypes
    vectorizer = CountVectorizerWrapper(columns_to_use="object")



Drop Unused Columns
-----------------
aikit transformer can also decided what you do with the columns you didn't encode.
By default most transformer drop those columns. That way at the end of the transformer you retrieve only the encoded columns.

The behavior is setted by the parameter 'drop_used_columns':
 * True : means you have only the encoded 'columns_to_use' result at the end
 * False : means you have the encoded 'columns_to_use' + the other columns (un-touched by the transformer)
 
This can make it easier to transformed part of the data.

Remark : the only transformers that have 'drop_used_columns = False' as default are categorical encoder. That way they automatically encoded  the categorical columns but keep the numerical column un-touched. Which means you can plug that at the begginning of your pipeline.


Drop Used Columns
-----------------
You can also decided if you want to keep the 'columns_to_use' in their original format (pre-encoding).
To do that you need to specify 'drop_used_columns=False'.
If you do that you'll have both encoded and non-encoded value after the transformers. This can be usefull sometimes.

For example, let's say that you want to do an SVD but you also want to keep the original columns (so the SVD is not reducing the dimension but adding new compositie features).
You can do it like that::

    from aikit.transformers import TruncatedSVDWrapper
    svd = TruncatedSVDWrapper(columns_to_use="all", n_components=5, drop_used_columns=False)

You can wrap your own model
---------------------------

You can use aikit wrapper for your own model, this is useful if you want to code a new transformer but you don't want to think about all the details to make it robust.

See :

.. toctree::
   :maxdepth: 1
   :titlesonly:
  
   model_wrapper_howto

Other Transformers
------------------

For a full Description of aikit Transformers go there :

.. toctree::
   :maxdepth: 1
   :titlesonly:
    
   all_transformers