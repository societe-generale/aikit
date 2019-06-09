![Build Status](https://travis-ci.org/societe-generale/aikit.svg?branch=master)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/societe-generale/aikit)
[![PyPI version](https://badge.fury.io/py/aikit.svg)](https://badge.fury.io/py/aikit)
[![Documentation Status](https://readthedocs.org/projects/aikit/badge/?version=latest)](https://aikit.readthedocs.io/en/latest/?badge=latest)

# aikit
Automatic Tool Kit for Machine Learning and Datascience.

The objective is to provide tools to ease the repetitive part of the DataScientist job and so that he/she can focus on modelization. This package is still in alpha and more features will be added.
Its mains features are:
 * improved and new "scikit-learn like" transformers ;
 * GraphPipeline : an extension of sklearn Pipeline that handles more generic chains of tranformations ;
 * an AutoML to automatically search throught several transformers and models.
 
 Full documentation is available here: https://aikit.readthedocs.io/en/latest/
 
### GraphPipeline

The GraphPipeline object is an extension of `sklearn.pipeline.Pipeline` but the transformers/models can be chained with any directed graph.

The objects takes as input two arguments:
 * models: dictionary of models (each key is the name of a given node, and each corresponding value is the transformer corresponding to that node)
 * edges: list of tuples that links the nodes to each other

Example:
```python
gpipeline = GraphPipeline(
    models = {
        "vect": CountVectorizerWrapper(analyzer="char",
                                       ngram_range=(1, 4),
                                       columns_to_use=["text1", "text2"]),
        "cat": NumericalEncoder(columns_to_use=["cat1", "cat2"]), 
        "rf": RandomForestClassifier(n_estimators=100)
    },
    edges = [("vect", "rf"), ("cat", "rf")]
)
```

![Alt text](docs/img/graphpipeline_mergingpipe.png?raw=true "Title")

### AutoML

Aikit contains an AutoML part which will test several models and transformers for a given dataset.

For example, you can create the following python script `run_automl_titanic.py`:
```python
from aikit.datasets import load_dataset, DatasetEnum
from aikit.ml_machine import MlMachineLauncher

def loader():
    dfX, y, *_ = load_dataset(DatasetEnum.titanic)
    return dfX, y

def set_configs(launcher):
    """ modify that function to change launcher configuration """
    launcher.job_config.score_base_line = 0.75
    launcher.job_config.allow_approx_cv = True
    return launcher

if __name__ == "__main__":
    launcher = MlMachineLauncher(base_folder = "~/automl/titanic", 
                                 name = "titanic",
                                 loader = loader,
                                 set_configs = set_configs)
    launcher.execute_processed_command_argument()
```

And then run the command:
```
python run_automl_titanic.py run -n 4
```

To run the automl using 4 workers, the results will be stored in the specified folder
You can aggregate those result using:
```
python run_automl_titanic.py result
```
