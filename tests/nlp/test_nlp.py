# Copyright 2026 Firefly Software Foundation.
"""Tests for the HuggingFace text classifier (integration: downloads a tiny model)."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_hf_text_classifier_runs() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from fireflyframework_datascience.nlp.adapters import HFTextClassifier

    texts = [
        "i love this",
        "great product",
        "amazing work",
        "best ever",
        "i hate this",
        "terrible thing",
        "awful result",
        "the worst",
    ]
    labels = ["pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"]

    model = HFTextClassifier(epochs=4).fit(texts, labels)
    preds = model.predict(["i really love it", "this is terrible"])
    assert len(preds) == 2
    assert set(model.classes) == {"pos", "neg"}
    assert all(p in {"pos", "neg"} for p in preds)
