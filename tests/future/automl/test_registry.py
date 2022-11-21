from sklearn.ensemble import RandomForestClassifier

from aikit.future.automl._hyper_parameters import HyperCrossProduct, HyperComposition
from aikit.future.automl._registry import MODEL_REGISTRY, get_init_parameters


def test_get_init_parameters():
    params = get_init_parameters(RandomForestClassifier)
    assert isinstance(params, dict)
    assert "self" not in params
    assert "n_estimators" in params


def test_model_registry():
    assert ("Model", "RandomForestClassifier") in MODEL_REGISTRY.hyper_parameters

    for key, value in MODEL_REGISTRY.hyper_parameters.items():
        assert isinstance(value, (HyperCrossProduct, HyperComposition))

    assert ("Model", "RandomForestClassifier") in MODEL_REGISTRY.default_hyper_parameters
    assert MODEL_REGISTRY \
        .default_hyper_parameters[("Model", "RandomForestClassifier")] \
        .get("random_state", None) is not None
