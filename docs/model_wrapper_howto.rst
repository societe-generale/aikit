
.. _model_wrapper_howto:


How to Wrap a transformer
=========================

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
                     columns_to_use = "all",
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
                     columns_to_use = "all",
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
==========================

To help understand a little more what goes on, here is a brief summary the fit method

 #. if 'columns_to_use' is set, creation and fit of a :class:`aikit.transformers.model_wrapper.ColumnsSelector` to subset the column
 #. type and shape of input are stored
 #. input is converted if it is not among the list of accepted input types
 #. input is converted to be 1 or 2 dimensions (also depending on what is accepted by the underlying transformer)
 #. underlying transformer is created (using '_get_model') and fitted
 #. logic is applied to try to figure out the features names
