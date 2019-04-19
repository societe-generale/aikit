
.. _block_manager:

Block Selector
--------------

In order to facilitate use-cases where several types of data are presented (as well as for internal use) two things were created 
 * a :class:`BlockSelector` selector transformer whose job is to select a given *block of data* (See after)
 * a :class:`BlockManager` object to store and retrieve blocks of Data.
 
Here is the explanation. Let's imagine you have a use case where data has more than one type. Example you have text and regular features, or text and image.
Typically to handle that you would need to put everyting into a DataFrame and then use selectors to retrieve different part of the data.
This is already greatly facilitated by aikit with the GraphPipeline, the fact that wrapped models can select the variables they work on and the fact that this selection can be done uses regex (and consequently prefix).
You could also deal with it manually but then cross-validation, splitting or train and test, would also have to be done manually.

However sometime it is just not praticle or possible to merge everyting into a DataFrame :
 * you might have a big sparse entry and other regular features : not praticle to merge everything into either a dense or sparse object
 * you might have a picture and regular feature and typically the pictures will be stored in a 3 or 4 dimensionals (observation x height x width x channels) tensor and not the classical 2 dimensions object
 * .... any other reason
 
What you can do is put ALLs your data into a dictionnary (or a BlockManager, see just after) : one key per type of data and one value per data.
You you pass that object to a block selector it can retrieve each block seperately::

    Xtrain = {"regular_features":pandas_dataframe_object,
              "sparse_features" :scipy_sparse_object
              }
              
    block_selector = BlockSelector("regular_features")
    df = block_selector.fit_transform(Xtrain)
    
Here *df* is just the *pandas_dataframe_object* object.
Remark : fit_transform is used as this work like a classical transformer. Even if the transformer doesn't do anything during the fit.

So you can just put those objects at the top of your pipeline and uses a dictionnary of data inplace of X.

The BlockSelector object also work with a list of datas (in that case the block to select is the integer corresponding to the position)::

    Xtrain = [pandas_dataframe_object,
              scipy_sparse_object]
              
    block_selector = BlockSelector(0)
    df = block_selector.fit_transform(Xtrain)

Example of such pipeline::

    GraphPipeline(models = {
            "sel_sparse":BlockSelector("sparse_features"),
            "svd":TruncatedSVDWrapper(),
            "sel_other":BlockSelector("regular_features"),
            "rf":RandomForestClassifier()
            },
    edges = [("sel_sparse","svd","rf"),("sel_other","rf")])

That model can be fitted however, you can't cross-validate it easily. That's is because Xtrain (which is a dictionnary of datas or a list of datas) isn't subsetable : you can't do Xtrain[index,:] or Xtrain.loc[index,:].

Block Manager
-------------
To solve this a new type of object is needed : the :class:`BlockManager`. This object is conceptually exactly like the Xtrain object before, it can either be a dictionnary of data or a list of data.
However it has a few additionnal things that allow it to work well within sklearn environnement.
 * it has a shape attribute
 * it can be subsetted using 'iloc'
 
Example::

    df = pd.DataFrame({"a":np.arange(10),"b":["aaa","bbb","ccc"] * 3 + ["ddd"]})
    arr = np.random.randn(df.shape[0],5)
    X = BlockManager({"df":df, "arr":arr})
    
    X["df"]        # this retrieves the DataFrame df (no copy)
    X["arr"]       # this retrieves the numpy array arr (no copy)
    X.iloc[0:5,:]  # this create a smaller BlockManager object with only 5 observations
    
    block_selector = BlockSelector("arr")
    block_selector.fit_transform()         #This retrieve the numpy array "arr" as well
    
    