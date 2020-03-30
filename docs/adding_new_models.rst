.. _adding_new_models

How to add new models
=====================

One of the idea of the package is to offer ways to quickly test ideas without much burden.
It is thus relatively easy to add new models/transformers to the framework.

Those models/transformers can be included in the search of the auto-ml component to be tested on different databases and with other transformers/models.

A model needs to be added at two different places in order to be fully integrated within the framework.

Let's see what's need to be done to include an hypothetic new models::

    class ReallyCoolNewTransformer(BaseEstimator, TransformerMixin):
        """ This is a great new  transformer """
        def __init__(self, super_choice):
            self.super_choice = super_choice
            
        def fit(self,X, y = None):
            pass
            
        def transform(self,X):
            pass


Add model to Simple register
----------------------------

This will allow the function 'sklearn_model_from_param' to be able to use your new model. The class simply needs to be added to the DICO_NAME_KLASS object::

    from aikit.model_definition import DICO_NAME_KLASS
    DICO_NAME_KLASS.add_klass(ReallyCoolNewTransformer)

Now that this is done, you can call the transformer by its *name*::

    from aikit.model_definition import sklearn_model_from_param
    model = sklearn_model_from_param(("ReallyCoolNewTransformer",{}))
    
model is an instance of ReallyCoolNewTransformer

Add model to Auto-Ml framework
------------------------------

This is a little more complicated, a few more informations need to be entered:

 * type of model
 * type of variable it uses
 * hyper-parameters
 
To do that you need to use the @register decorator::

    from aikit.ml_machine.ml_machine_registration import register, _AbstractModelRepresentationDefault, StepCategories
    import aikit.ml_machine.hyper_parameters as hp

    @register
    class DimensionReduction_ReallyCoolNewTransformer(_AbstractModelRepresentationDefault):
        klass = ReallyCoolNewTransformer
        
        category = StepCategories.DimensionReduction
        type_of_variable = None
        type_of_model = None # Used for all problem
        
        custom_hyper = {"super_parameters":hp.HyperChoice(("superchoice_a","superchoice_b"))}

    
See :ref:`model_register` for complete description of register.
See :ref:`hyper_parameters` for complete description of register.

Remark:
The registers behaves like singletons so you can modify them in any part of the code.
You just need the code to be executed somewhere for it to work.

If a model is stable and tested enough the new entry can be added to the python files :

 * 'model_definition.py' : for the simple register
 * ml_machine/ml_machine_registration.py : for the full auto-ml register

(See :ref:`contribution` for detailed about how to contribute to the evolution of the library)

Remark : you don't need to use the wrapper for your model to be incorporated in the framework. However, it is best to do so. That way you can focus on the logic and let the wrapper make your model more generic.


.. toctree::
   :maxdepth: 1
   :hidden:

   model_register
   hyper_parameters


