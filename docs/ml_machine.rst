.. _ml_machine:

Ml Machine
==========

Here is a more detailed explanation about what the Ml Machine is doing.


Steps
-----
Each transformation are grouped in steps. There can be several steps needed for each DataFrame, and the needed steps depend on the type of problem/variable.
Some steps are optional some are needed.

Example of such steps:

 * TextPreprocessing (optional) this steps has all the text preprocessing transformer
 * TextEncoder : (needed) this step encodes the text into numerical value (Example using CountVectorizer or Word2Vec)
 
 * MissingValueImputer (needed if some variable have missing values)
 * CategorieEncoder (needed if some variables are categorical)
 * ...
 * Model (needed) : last step consisting of the prediction model
 
see complete list with :class:`aikit.enums.StepCategories`

The Ml Machine will randomly draw one model per step and merge them into a complex processing pipeline.
Optional steps are sometimes drawn and sometime not.

(The transformers that are drawn are the one in the ml machine registry : :ref:`ml_machine_registration`)

First Rounds
------------

Before randomly selected pipelines, a first round of models are tested using:
 * default parameters
 * without all the optional steps
 
Usually those pipelines should perform *relatively* well and gives a good idea about what work and what doesn't.


Next Rounds
-----------
Once that is done, random rounds are started. For those, random models are drawn:

 * for each step, draw a random transformation (or ignore the step if it is optional)
 * for each model draw random hyper-parameters
 * if block of variable were setted, randomly draw a subset of those blocks
 * merge everythihg into a complexe graph
 
That model is then *send* to the worker to be cross-validated.

Stopping Threshold
------------------
For each given model the worker aims to do a full cross-validation. However the cross-validation can be stopped after the first fold if the result are too low (bellow a threshold fixed by the controller).

That threshold is computed using :
 * the base line score if it exists
 * a quantile on already done result
 
(See :func:`aikit.cross_validation.cross_validation` which is used to compute the cross-validation)

Guided Job
----------
After a few models, with a given random probability, the controller will start to create *Guided Jobs*. Those jobs are not random anymore but uses BayesianOptimization to try to guess a model that will perform correctly.

Concretely a *meta model* is fitted to try to predict performance based on hyper-paramters and transformers/models choices. And instead we use that meta model to predict wheither or not a candidate model will perform or not.


Random Model Generator
----------------------
The random model generator can be used outside of the Ml Machine::

    from aikit.ml_machine.ml_machine import RandomModelGenerator
    generator = RandomModelGenerator( auto_ml_config = auto_ml_config)
    
    Graph, all_models_params, block_to_use = generator.draw_random_graph()
    
The generator returns three things:
 1. Graph : networkx graph of the model
 2. all_models_params : dictionnary with all the hyper-parameters of all the transformers/models
 3. block_to_use : the block of columns to use
    

With this 3 objects the json of a model can be created::

    from aikit.ml_machine.model_graph import convert_graph_to_code
    model_json_code = convert_graph_to_code(Graph, all_models_params
    
And then a working model can be created::

    from aikit.model_definition import sklearn_model_from_param
    skmodel = aikit.model_definition.sklearn_model_from_param(model_json_code)

    
