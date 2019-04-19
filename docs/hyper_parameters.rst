.. _hyper_parameters:

Hyperparameters
===============

Here you'll find the explanation of how to specify random hyper-parameters.
Those hyper-parameters are used to generate random values of the parameters of a given model.

Example, to generate a random integer between 1 and 10 (included) you can use HyperRangeInt::

    hyper = HyperRangeInt(start = 1, end = 10)
    [hyper.get_rand() for _ in range(5)]
    
    >> [4,1,10,3,3]
 
The complete list of HyperParameters are available here :module:`aikit.ml_machine.hyper_parameters`
Each class implements a 'get_rand' method.

HyperCrossProduct
-----------------
This class is used to combine hyper-parameters together which is needed to generate complexe dictionnary-like hyper-parameters

 .. autoclass:: aikit.ml_machine.hyper_parameters.HyperCrossProduct

Example::

    hp = HyperCrossProduct({"int_value":HyperRangeInt(0,10), "float_value":HyperRangeFloat(0,1)})
    hp.get_rand()
    
This will generate random dictionnary with keys 'int_value' and 'float_value'


    
HyperComposition
----------------
This class is used to include dependency between parameters or to create an hyper parameters from two different distributions


 .. autoclass:: aikit.ml_machine.hyper_parameters.HyperComposition
 
Example::

    hp = HyperComposition([ (0.9,HyperRangeInt(0,100)) ,(0.1,HyperRangeInt(100,1000)) ])
    hp.get_rand()
    
This will generate a random number between 0 and 100 with probability 0.9 and one between 100 and 1000 with probability 0.1

::

    hp = HyperComposition([
        (0.5 , HyperComposition({"analyzer":"char" , "n_gram_range":[1,4]})),
        (0.5 , HyperComposition({"analyzer":"word" , "n_gram_range":1}) )
    
    ])
    hp.get_rand()
    
This will generate with probability:

 * 1/2 a dictionnary with "analyzer":"char" and "n_gram_range": random between 1 and 4
 * 1/2 a dictionnary with "analyzer":"word" and "n_gram_range": 1
