"""
Regression test for the freeze/unfreeze logic in model_factory.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from model_factory import build_model, unfreeze_head_only, unfreeze_partial, count_trainable_params

def test_resnet50_head_only_trainable_fraction_is_small():
    model = build_model("resnet50")
    unfreeze_head_only(model, "resnet50")
    trainable, total = count_trainable_params(model)
    fraction = trainable / total
    assert fraction < 0.05, f"Expected <5% trainable in head-only phase, got {fraction:.1%}"

def test_resnet50_partial_trainable_fraction_in_expected_range():
    model = build_model("resnet50")
    unfreeze_partial(model, "resnet50")
    trainable, total = count_trainable_params(model)
    fraction = trainable / total
    assert 0.85 < fraction < 1.0, (
        f"Expected ~90-95% trainable in ResNet50 partial fine-tune (layer3+layer4+fc), got {fraction:.1%}")

def test_vit_head_only_trainable_fraction_is_small():
    model = build_model("vit_b_16")
    unfreeze_head_only(model, "vit_b_16")
    trainable, total = count_trainable_params(model)
    fraction = trainable / total
    assert fraction < 0.05, f"Expected <5% trainable in head-only phase, got {fraction:.1%}"

def test_vit_partial_trainable_fraction_in_expected_range():
    model = build_model("vit_b_16")
    unfreeze_partial(model, "vit_b_16")
    trainable, total = count_trainable_params(model)
    fraction = trainable / total
    assert 0.25 < fraction < 0.40, (
        f"Expected ~33% trainable in ViT partial fine-tune, got {fraction:.1%}"
    )