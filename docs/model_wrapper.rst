
.. _model_wrapper:

ModelWrapper
============

This is a more detailed explanation about the :class:`aikit.transformers.model_wrapper.ModelWrapper` class. It will explain what the wrapper is doing and how to wrap new models.

Wrapper Goal
------------
The aim of the wrapper is to provide a generic class to handle most of the redondant operations that we might want to apply in a transformer.
In particular it aims at making regular 'sklearn like' model more generic and more 'user friendly'.

Here are a few reasons wrapping a transfomer might be useful :

 * automatic conversion of input/output into a given format (which is useful when chaining models and some on them accepts DataFrame, some don't, ...)
 * verification of type, shape of new data
 * shape conversion for model that only accept '1-dimensional' input 
 * automatic splits and concatenation of result for models that only work one column at a time (See : :ref:`CountVectorizerWrapper`)
 * generation of features_names and usage of those names when the output is a DataFrame
 * delay the creation of underlying model until the :func:`fit` is called. This allow to customize hyper-parameters based on the data (Ex : 'n_components' can be a float, corresponding to the fraction of columns to keep)

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

How to Wrap a transformer
-------------------------

To wrap a new model you should 
 1. Create a new class that inherit from ModelWrapper
 2. In the __init__ of that class specify the *rules* of the wrapper (see just after)
 3. create a _get_model method to specify the underlying transformers

 .. autoclass:: aikit.transformers.model_wrapper.ModelWrapper
 
A few notes:
 * must_transform_to_get_features_name and dont_change_columns are here to help the wrapped transformers to implement a correct 'get_feature_names'
 * the wrapped model has a 'model' attribute that retrieves the underlying transformer(s)
 * the wrapped model will generate a NotFittedError error when called without being fit first (this behavior is not consistent across all transformers)
 
 
Here is an example of how to wrap sklearn CountVectorizer::

    class CountVectorizerWrapper(ModelWrapper):
        """ wrapper around sklearn CountVectorizer with additionnal capabilities
        
         * can select its columns to keep/drop
         * work on more than one columns
         * can return a DataFrame
         * can add a prefix to the name of columns

        """
        def __init__(self,
                     analyzer = "word",
                     max_df = 1.0,
                     min_df = 1,
                     ngram_range = 1,
                     max_features = None,
                     columns_to_use = None,
                     regex_match    = False,
                     desired_output_type = DataTypes.SparseArray
                     ):
            
            self.analyzer = analyzer
            self.max_df = max_df
            self.min_df = min_df
            self.ngram_range = ngram_range
            self.columns_to_use = columns_to_use
            self.regex_match    = regex_match
            self.desired_output_type = desired_output_type 
            
            super(CountVectorizerWrapper,self).__init__(
                columns_to_use = columns_to_use,
                regex_match    = regex_match,
                
                work_on_one_column_only = True,
                all_columns_at_once = False,
                accepted_input_types = (DataTypes.DataFrame,DataTypes.NumpyArray),
                column_prefix = "BAG",
                desired_output_type = desired_output_type,
                must_transform_to_get_features_name = False,
                dont_change_columns = False)
            
            
        def _get_model(self,X,y = None):
            
            if not isinstance(self.ngram_range,(tuple,list)):
                ngram_range = (1,self.ngram_range)
            else:
                ngram_range = self.ngram_range
                
            ngram_range = tuple(ngram_range)  
            
            return CountVectorizer(analyzer = self.analyzer,
                                   max_df = self.max_df,
                                   min_df = self.min_df,
                                   ngram_range = ngram_range)
                                   
And here is an example of how to wrap TruncatedSVD to make it work with DataFrame and create columns features::

    class TruncatedSVDWrapper(ModelWrapper):
        """ wrapper around sklearn TruncatedSVD 
        
        * can select its columns to keep/drop
        * work on more than one columns
        * can return a DataFrame
        * can add a prefix to the name of columns
        
        n_components can be a float, if that is the case it is considered to be a percentage of the total number of columns
        
        """
        def __init__(self,
                     n_components = 2,
                     columns_to_use = None,
                     regex_match  = False
                     ):
            self.n_components = n_components
            self.columns_to_use = columns_to_use
            self.regex_match    = regex_match
            
            super(TruncatedSVDWrapper,self).__init__(
                columns_to_use = columns_to_use,
                regex_match    = regex_match,
                
                work_on_one_column_only = False,
                all_columns_at_once = True,
                accepted_input_types = None,
                column_prefix = "SVD",
                desired_output_type = DataTypes.DataFrame,
                must_transform_to_get_features_name = True,
                dont_change_columns = False)
            
            
        def _get_model(self,X,y = None):
            
            nbcolumns = _nbcols(X)
            n_components = int_n_components(nbcolumns, self.n_components)
            
            return TruncatedSVD(n_components = n_components)

            
What append during the fit
--------------------------
To help understand a little more what goes on, here is a brief summary the fit method

 #. if 'columns_to_use' is set, creation and fit of a :class:`aikit.transformers.model_wrapper.ColumnsSelector` to subset the column
 #. type and shape of input are stored
 #. input is converted if it is not among the list of accepted input types
 #. input is converted to be 1 or 2 dimensions (also depending on what is accepted by the underlying transformer)
 #. underlying transformer is created (using '_get_model') and fitted
 #. logic is applied to try to figure out the features names
