from src.deep_learning import TrainingConfig
from src.nlp import DEFAULT_SENTIMENT_MODEL


def test_training_config_is_explicit_and_reproducible():
    config = TrainingConfig()
    assert config.random_state == 42
    assert config.threshold == 0.5
    assert config.hidden_dims == (256,128, 64, 32)
    assert DEFAULT_SENTIMENT_MODEL.startswith("distilbert/")
