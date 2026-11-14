"""Reusable evaluation dataset loader, validator, and schema definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class EvaluationSample:
    """A single evaluation query case with ground truth references."""

    id: str
    question: str
    expected_answer: str
    relevant_docs: List[str] = field(default_factory=list)
    expected_context_snippets: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationSample:
        if not data.get("id"):
            raise ValueError("Evaluation sample must have an 'id'.")
        if not data.get("question"):
            raise ValueError("Evaluation sample must have a non-empty 'question'.")
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            expected_answer=str(data.get("expected_answer", "")),
            relevant_docs=[str(d) for d in data.get("relevant_docs", [])],
            expected_context_snippets=[str(s) for s in data.get("expected_context_snippets", [])],
            relevant_chunk_ids=[str(cid) for cid in data.get("relevant_chunk_ids", [])],
            category=str(data.get("category", "general")),
            metadata=dict(data.get("metadata", {})),
        )


class EvaluationDataset:
    """Manages an evaluation benchmark dataset."""

    DEFAULT_DATASET_PATH = Path("data/evaluation/default_eval_dataset.json")

    def __init__(
        self,
        samples: Optional[List[EvaluationSample]] = None,
        name: str = "Custom Evaluation Dataset",
        description: str = "",
        version: str = "1.0",
    ):
        self.samples: List[EvaluationSample] = samples or []
        self.name = name
        self.description = description
        self.version = version

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EvaluationSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[EvaluationSample]:
        return iter(self.samples)

    def add_sample(self, sample: EvaluationSample) -> None:
        self.samples.append(sample)

    def filter_by_category(self, category: str) -> EvaluationDataset:
        filtered = [s for s in self.samples if s.category.lower() == category.lower()]
        return EvaluationDataset(
            samples=filtered,
            name=f"{self.name} ({category})",
            description=self.description,
            version=self.version,
        )

    def get_categories(self) -> List[str]:
        return sorted(list({s.category for s in self.samples}))

    def summary(self) -> Dict[str, Any]:
        categories: Dict[str, int] = {}
        for s in self.samples:
            categories[s.category] = categories.get(s.category, 0) + 1
        return {
            "name": self.name,
            "version": self.version,
            "total_samples": len(self.samples),
            "categories": categories,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "samples": [s.to_dict() for s in self.samples],
        }

    def save_to_file(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Optional[str | Path] = None) -> EvaluationDataset:
        target_path = Path(file_path) if file_path else cls.DEFAULT_DATASET_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.load_from_dict(data)

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> EvaluationDataset:
        samples_raw = data.get("samples", [])
        samples = [EvaluationSample.from_dict(item) for item in samples_raw]
        return cls(
            samples=samples,
            name=data.get("name", "Loaded Evaluation Dataset"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )
