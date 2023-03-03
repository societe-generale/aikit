import networkx as nx
import pandas as pd

from ._config import AutoMlConfig
from .random_model_generator import RandomModelGenerator
from ..graph import convert_graph_to_code
from ..util import CLASS_REGISTRY
from ..util.serialization import sklearn_model_from_param


def get_default_pipeline(X, y, final_model=None):  # noqa
    """ create a default GraphPipeline for a given model

    Parameters
    ----------
    X : pd.DataFrame
        the training data

    y : array like
        the target

    final_model : None or model instance
        if not None the model at the end the pipeline to use

    Returns
    -------
    a full pipeline to be fitted
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)  # noqa

    automl_config = AutoMlConfig(X, y)
    automl_config.guess_everything(X, y)

    if ('Model', 'RandomForestClassifier') in automl_config.models_to_keep:
        automl_config.filter_models(Model='RandomForestClassifier')
    else:
        automl_config.filter_models(Model='RandomForestRegressor')

    if len([m for m in automl_config.models_to_keep if m[0] == "Model"]) != 1:
        raise ValueError("No default model found.")

    generator = RandomModelGenerator(automl_config=automl_config)
    iterator = generator.default_models_iterator()

    graph, all_models_params, blocks_to_use = next(iterator)    # Retrieve first default model

    if final_model is not None:
        if not hasattr(final_model, "fit"):
            raise ValueError("'final_model' should have a 'fit' method")
        # Modify existing pipeline final model with specified final_model
        node = None
        for node in graph.nodes:
            if node[0] == 'Model':
                break
        assert node is not None
        new_node = ('Model', ('Model', final_model.__class__.__name__))

        graph = nx.relabel_nodes(graph, {node: new_node})
        del all_models_params[node]
        all_models_params[new_node] = final_model.get_params()

        CLASS_REGISTRY.add_klass(final_model.__class__)

    json_code = convert_graph_to_code(graph, all_models_params, return_mapping=True)

    model = sklearn_model_from_param(json_code["json_code"])

    return model
