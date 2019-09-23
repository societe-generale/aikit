.. _ml_machine_launcher_advanced:

Advanced functionnalities
-------------------------

The launcher script can be used to specify lots of things in the ml machine. The auto-ml makes lots of choices by default which can be changed.
To change those you need to modify the 'job_config' object or the 'auto_ml_config' object.

Within the launcher, you can use the 'set_configs' function to do just that.

Here is a complete example::

    def set_configs(launcher):
        """ this is the function that will set the different configurations """
        # Change the CV here :
        launcher.job_config.cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123) # specify CV
        
        # Change the scorer to use :
        launcher.job_config.scoring = ['accuracy', 'log_loss_patched', 'avg_roc_auc', 'f1_macro']
        
        # Change the main scorer (linked with )
        launcher.job_config.main_scorer = 'accuracy'
        
        # Change the base line (for the main scorer)
        launcher.job_config.score_base_line = 0.8
        
        # Allow 'approx cv or not : 
        launcher.job_config.allow_approx_cv = False
        
        # Allow 'block search' or not :
        launcher.job_config.do_blocks_search = True
        
        # Start with default models or not :
        launcher.job_config.start_with_default = True
        
        # Change default 'columns block' : use for block search
        launcher.auto_ml_config.columns_block = OrderedDict([
              ('NUM', ['pclass', 'age', 'sibsp', 'parch', 'fare', 'body']),
             ('TEXT', ['name', 'ticket']),
             ('CAT', ['sex', 'cabin', 'embarked', 'boat', 'home_dest'])])
    
        # Change the list of models/transformers to use :
        launcher.auto_ml_config.models_to_keep = [
                     #('Model', 'LogisticRegression'),
                     ('Model', 'RandomForestClassifier'),
                     #('Model', 'ExtraTreesClassifier'),
                     
                     # Example : keeping only RandomForestClassifer
                     
                     ('FeatureSelection', 'FeaturesSelectorClassifier'),
                     
                     ('TextEncoder', 'CountVectorizerWrapper'),
                     
                     #('TextPreprocessing', 'TextNltkProcessing'),
                     #('TextPreprocessing', 'TextDefaultProcessing'),
                     #('TextPreprocessing', 'TextDigitAnonymizer'),
                     
                     # => Example: removing TextPreprocessing
                     
                     ('CategoryEncoder', 'NumericalEncoder'),
                     ('CategoryEncoder', 'TargetEncoderClassifier'),
                     
                     ('MissingValueImputer', 'NumImputer'),
                     
                     ('DimensionReduction', 'TruncatedSVDWrapper'),
                     ('DimensionReduction', 'PCAWrapper'),
                     
                     ('TextDimensionReduction', 'TruncatedSVDWrapper'),
                     ('DimensionReduction', 'KMeansTransformer'),
                     ('Scaling', 'CdfScaler')
                     ]
        
        # Specify the type of problem
        launcher.auto_ml_config.type_of_problem = 'CLASSIFICATION'
        
        # Specify special hyper parameters : Example 
        launcher.auto_ml_config.specific_hyper = {
                ('Model', 'RandomForestClassifier') : {"n_estimators":[10,20]}
                }
        # Example : only test n_estimators to be 10 or 20
    
        return launcher

Job Config Option
*****************
The 'job_config' object stores information related to the way we will test the models : like the Cross-Validation object to use, the base-line, ...

1. Cross-Validation

You can change the 'cv' to use. Simply change the 'cv' attribute.

Remark : if you specify 'cv = 10', the type of CV will be guessed (StratifiedKFold, KFold, ...)

Remark : you can use a special 'RandomTrainTestCv' or 'IndexTrainTestCv' object to do only a 'train/test' split and not a full cross-validation.

2. Scorings

Here you can change the scorers to be used. Simplify specify a list with the name of scorers. You can also directly pass sklearn scorers.

3. Main Scorer

This specify the main scorer, it is used to compute the base-line. It can also be used to 'guide' the auto-ml.

4. Approximate CV

If you want to allow that or not. The idea of 'approximate cv' is to gain time by by-passing the CV of some transformers : if a transformers doesn't depend on the target you can reasonably skip cross-validation without having much leakage.

5. Do Block Search

If True, the auto-ml will do special jobs where the model is fixed, the preprocessing pipeline is the default one, but it tries to remove some of the block of columns. It also tries to use only one block.
Those jobs helps figure out what block of features are important or not.

6. Starts with default

If True, the auto-ml starts by 'default' models and transformers before doing the full bayesian random search.

Auto Ml Config
**************
The 'auto_ml_config' object stores the information related to the data, the problem, ...

1. Columns Blocks

Using that attribute you can change the blocks of columns, by default the blocks corresponds to the type of variable (Numerical, Categories and Text) but you can specify what you want.
Those blocks will be used for the 'block search' jobs.

2. Models to Keep

Here you can filter the models/transformers to test.

Remark : you need to keep required preprocessing steps. For example, if you have text columns you need to keep at least one text encoder.

3. Type of Problem

You can chage the type of problem. This is needed if the guessing was wrong.

4. Specific Hyper Parameters

You can change the hyper parameters used, simply pass a dictionnary with keys being the models to change, and values the new hyper-parameters.
The new hyper-parameters can either be a dict (as in the example above) or an object of the HyperCrossProduct class.

Usage of groups
***************
Sometime your data falls into different groups. Sklearn allow you to pass those information to the cross-validation object to make sure the folds respect the groups. Aikit also allow you to use those groups for custom scorer.
To use groups in the auto-ml the 'loader' function needs to returns three things instead of two : 'dfX, y, groups'

You can then specify a special CV or a special scorer that uses the groups.
