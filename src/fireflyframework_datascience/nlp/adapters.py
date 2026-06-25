# Copyright 2026 Firefly Software Foundation.
"""HuggingFace text-classification adapter (requires the ``nlp`` extra: transformers + torch)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fireflyframework_datascience.core.exceptions import AdapterUnavailableError
from fireflyframework_datascience.nlp import TextModel


class HFTextClassifier:
    """Fine-tunes a HuggingFace sequence-classification model on ``(texts, labels)``.

    Defaults to DistilBERT (a small, fast-tokenizer checkpoint); swap ``model_name`` for any other
    sequence-classification model (RoBERTa, DeBERTa, …). This is the NLP modality on the same
    fit/predict contract as the tabular trainers; vision follows the same shape in ``vision.adapters``.
    """

    name = "hf_text"

    def __init__(
        self,
        *,
        model_name: str = "distilbert-base-uncased",
        epochs: int = 3,
        lr: float = 5e-5,
        max_length: int = 64,
        batch_size: int = 8,
    ) -> None:
        self._model_name = model_name
        self._epochs = epochs
        self._lr = lr
        self._max_length = max_length
        self._batch_size = batch_size

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> TextModel:
        try:
            import torch  # type: ignore[import-not-found, import-untyped]
            from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found, import-untyped]
            from transformers import (  # type: ignore[import-not-found, import-untyped]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise AdapterUnavailableError("HFTextClassifier", "nlp") from exc

        classes = sorted(set(labels), key=str)
        label_to_id = {label: i for i, label in enumerate(classes)}
        y = torch.tensor([label_to_id[label] for label in labels], dtype=torch.long)

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        encoded = tokenizer(
            list(texts), truncation=True, padding=True, max_length=self._max_length, return_tensors="pt"
        )
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name, num_labels=len(classes))

        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], y)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr)

        model.train()
        for _ in range(self._epochs):
            for input_ids, attention_mask, labels_batch in loader:
                optimizer.zero_grad()
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                output.loss.backward()
                optimizer.step()
        model.eval()
        return TextModel("hf_text", _HFPredictor(tokenizer, model, classes, self._max_length), classes)


class _HFPredictor:
    """Tokenizes inputs and maps model logits back to the original labels."""

    def __init__(self, tokenizer: Any, model: Any, classes: list[Any], max_length: int) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._classes = classes
        self._max_length = max_length

    def predict(self, texts: Sequence[str]) -> list[Any]:
        import torch  # type: ignore[import-not-found, import-untyped]

        encoded = self._tokenizer(
            list(texts), truncation=True, padding=True, max_length=self._max_length, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self._model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
        return [self._classes[i] for i in logits.argmax(dim=1).tolist()]


__all__ = ["HFTextClassifier"]
