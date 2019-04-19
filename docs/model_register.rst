
.. _model_register:

Model Register
==============

To be able to randomly create complex processing pipelines the Ml Machine needs to know a few things.

 1. The list of models/transformers to use
 2. Information about each models/transformations
 
For each models/transformers here are the information needed:
 1. the step on which is will be used  : CategorieEncoder, TextProcessing, ...
 2. the type of variable it will be used on : Text, Numerical, Categorie or None (for everything)
 3. can it be used for regression, classification or both
 4. the list of hyper-parameters and their values
 5. ... any other needed piece of information
 
To be able to that each transformers and models should be register and everything is stored into an object.

All that is done within :mod:`aikit.ml_machine.ml_machine_registration`

To register a new model simply create a class and decorate it::

    @register
    class RandomForestClassifier_Model(_AbstractModelRepresentationDefault):
        
        klass = RandomForestClassifier
        category = StepCategories.Model
        type_of_variable = None
        
        custom_hyper = {"criterion": ("gini","entropy") }
        
        is_regression = False
        
        
        default_parameters = {"n_estimators":100}
        
        
Remark : the name of the class doesn't matter, no instance will ever be created. It is just a nice way to write information.

A few things are needed within that class:

* klass : should contain the actual class of the underlying model
* category : one of the StepCategories choice
* type_of_variable : if None it means that the model should be applied to the complete Dataset (this field will be used to create branches with the different type of variable in the pipeline)
* is_regression : True, False or None (None means both)
* default_parameters : to override the klass default parameters. Those parameters are use during the First Round of the Ml Machine

HyperParameters
---------------

Each class should be able to generate its hyper-parameters, that is done by default with the 'get_hyper_parameter' class method.

Here is what is done to generate the hyper-parameters:

For each parameters within the **signature** of the __init__ method of the klass:
 * if the parameters is present within 'custom_hyper', use that to generate the corresponding values
 * if it is present within the 'default_hyper', use that to generate the corresponding values
 * if not, don't include it in the hyperparameters (and consequently the default value will be kept).
 

Default Hyper
-------------

Here is the list of the default hyper-parameters::

    default_hyper = {
        "n_components" : hp.HyperRangeFloat(start = 0.1,end = 1,step = 0.05),

        # Forest like estimators 
        "n_estimators" : hp.HyperComposition(
                            [(0.75,hp.HyperRangeInt(start = 25, end = 175, step = 25)),
                             (0.25,hp.HyperRangeInt(start = 200, end = 1000, step = 100))]),
    
        "max_features":hp.HyperComposition([ ( 0.25 , ["sqrt","auto"]),
                                             ( 0.75 , hp.HyperRangeBetaFloat(start = 0, end = 1,alpha = 3,beta = 1) )
                                           ]),
                       
        "max_depth":hp.HyperChoice([None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,50,100]),
        "min_samples_split":hp.HyperRangeBetaInt(start = 1,end = 100, alpha = 1, beta = 5),
                                 
        # Linear model
        "C":hp.HyperLogRangeFloat(start = 0.00001, end = 10, n = 50),

        "alpha":hp.HyperLogRangeFloat(start = 0.00001,end = 10, n = 50),

        # CV
        "analyzer":hp.HyperChoice(['word','char','char_wb']),
     }
     
This helps when hyper-parameters are common across many models.

Special HyperParameters
-----------------------

In some cases, hyper-parameters can't be drawn independently from each other. 

For example, by default, we might want to test either for a CountVectorizer:
 * 'char' encoding with bag of char of size 1,2,3 or 4
 * 'word' encoding only with bag of word of size 1
 
In that case we need to create a custom global hyper-parameter, that can be done by overriding the :func:`get_hyper_parameter` classmethod.

Example::

    @register
    class CountVectorizer_TextEncoder(_AbstractModelRepresentationDefault):
        klass = CountVectorizerWrapper
        category = StepCategories.TextEncoder
        type_of_variable = TypeOfVariables.TEXT
        
        
        @classmethod
        def get_hyper_parameter(cls):
            ### Specific function to handle the fact that I don't want ngram != 1 IF analyzer = word ###
            res = hp.HyperComposition([(0.5 , hp.HyperCrossProduct({"ngram_range":1,
                                                                   "analyzer":"word",
                                                                   "min_df":[1,0.001, 0.01, 0.05],
                                                                   "max_df":[0.999, 0.99, 0.95]
                                                                   }) ),
                                      (0.5  , hp.HyperCrossProduct({
                                                  "ngram_range": hp.HyperRangeInt(start = 1,end = 4),
                                                   "analyzer": hp.HyperChoice(("char","char_wb")) ,
                                                   "min_df":[1, 0.001, 0.01, 0.05],
                                                   "max_df":[0.999, 0.99, 0.95]
                                                   }) )
                                        ])
                                                             

            
            return res

This tells that the global hyperparameter is a composition between bag of word with ngram_range = 1 and bag of char with ngram between 1 and 4

(See :ref:`hyper_parameters` for detailed explanation on how to specify hyper-parameters)



