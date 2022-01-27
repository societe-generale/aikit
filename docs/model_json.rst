
.. _model_json:

ModelJson
=========


Model representation
--------------------

It is sometime useful to specify the model to use for a given use-case outside of the main code of the project. For example in a json like object.
This can have several advantages :
 * allow the change of the underlying model without change any code (example : shift from a :class:`RandomForestClassifier` to a :class:`LGBMClassifier`)
 * allow the same code to be used for different sub problem *BUT* allowing specific hyper-parameters/models for each sub problems
 * easier to incorporate model that were found automatically by an :ref:`ml_machine`

To be able to do that we need to save the description of a complex model into a simple json like format.

The syntax is easy : a model is represented by a tuple with its name and its hyper-parameters.

Example, the model::

    RandomForestClassifier(n_estimators=100)
    
is represented by the object::

    ("RandomForestClassifier",{"n_estimators":100})

So : klass(**kwargs) is equivalent to ('klass',kwargs)

Let's take a more complexe example using a GraphPipeline::

    gpipeline = GraphPipeline(models = {"vect" : CountVectorizerWrapper(analyzer="char",ngram_range=(1,4)),
                                            "svd"  : TruncatedSVDWrapper(n_components=400) ,
                                            "logit" : LogisticRegression(class_weight="balanced")},
                                   edges = [("vect","svd","logit")]
                                   )

is represented by::

    json_object = ("GraphPipeline", {"models": {"vect" : ("CountVectorizerWrapper"  , {"analyzer":"char","ngram_range":(1,4)} ),
                                 "svd"  : ("TruncatedSVDWrapper"     , {"n_components":400}) ,
                                 "logit": ("LogisticRegression" , {"class_weight":"balanced"}) },
                      "edges":[("vect","svd","logit")]
                      })                               

So if a given model uses other models as parameters it works as well.

Model conversion
----------------

Once the object is create you can convert it to a real (unfitted) model using :func:`aikit.model_definition.sklearn_model_from_param` ::

    sklearn_model_from_param(json_object)
    
which gives a model that can be fitted.


Json saving
-----------
That representation uses only simple types that are json serializable (string, number, list, dictionnary) and the json can be saved on disk.

Remark : since json doesn't allow :
 * tuple (only list are known)
 * dictionnary with non string keys
 
it is best to overried the json serializer to handle those type. The special encoder is found in :module:`aikit.tools.json_helper` and 'save_json' and 'load_json' can be used directly

Example saving the 'json_object' above::

    from aikit.tools.json_helper import save_json
    save_json(json_object, fname ="model.json")
    
    realoaded_json_object = load_json("model.json")
    
The special serializer works by transforming un-handle type into a dictionnary with
 * a '__items__' key with a list of object
 * a '__type__'  key with the original type
 
Example::

    ("a","b")
    
is transformed into::

    {"__items__":["a","b"], "__type__":"__tuple__"}
    
The handle types are :
 * dict : '__dict__'
 * tuple
   
Model Register
--------------
To be able to use a given model using only its name all the models should be registred in a dictionnary.

This is done within :mod:`aikit.simple_model_registration`, in that file you have a DICO_NAME_KLASS object which stored the classes of every model.
To add a new model simple use the add_klass method.

Example::

    DICO_NAME_KLASS.add_klass(LGBMClassifier)
    DICO_NAME_KLASS.add_klass(LGBMRegressor)
    
Remark : this registrer is different from the one used for the automatic machine learning part (:ref:`ml_machine`) which contain more informations (hyper-parameters, type, ...)


    







