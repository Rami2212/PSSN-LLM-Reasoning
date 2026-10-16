import json

import pytest
import torch

from src.black_hole.config import BlackHoleDetectorConfig
from src.black_hole.detector import BlackHoleDetector
from src.black_hole.loss import BlackHoleLoss, positive_class_weight


def test_detector_returns_one_probability_per_sentence_on_cpu():
    model = BlackHoleDetector(BlackHoleDetectorConfig(input_size=8, hidden_size=4, dropout=0.0))
    features = torch.randn(5, 8)

    probabilities = model(features)

    assert probabilities.shape == (5,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.equal(probabilities, model.predict_proba(features))


def test_detector_accepts_single_vector_and_rejects_wrong_width():
    model = BlackHoleDetector(BlackHoleDetectorConfig(input_size=8, hidden_size=4, dropout=0.0))
    assert model(torch.randn(8)).ndim == 0
    with pytest.raises(ValueError, match="Expected 8 features"):
        model(torch.randn(2, 7))


def test_configuration_round_trips_json(tmp_path):
    config = BlackHoleDetectorConfig(input_size=32, hidden_size=12, dropout=0.1)
    path = tmp_path / "detector_config.json"
    config.save(path)
    loaded = BlackHoleDetectorConfig.load(path)
    assert loaded == config
    assert json.loads(path.read_text(encoding="utf-8"))["input_size"] == 32


def test_weighted_loss_and_derived_class_weight():
    weight = positive_class_weight(torch.tensor([0, 0, 0, 1]))
    assert weight == pytest.approx(3.0)
    loss = BlackHoleLoss(positive_weight=weight)
    value = loss(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    assert torch.isfinite(value)


def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError, match="dropout"):
        BlackHoleDetectorConfig(input_size=8, hidden_size=4, dropout=1.0)
