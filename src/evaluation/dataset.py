import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalSample:
    question: str
    ground_truth_answer: str
    ground_truth_sources: list[str]
    metadata: dict[str, str]


class EvaluationDataset:
    def __init__(self, samples: list[EvalSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def from_json(cls, path: str) -> "EvaluationDataset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        samples = [
            EvalSample(
                question=item["question"],
                ground_truth_answer=item["ground_truth_answer"],
                ground_truth_sources=item.get("ground_truth_sources", []),
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
        return cls(samples)

    def to_json(self, path: str) -> None:
        data = [
            {
                "question": s.question,
                "ground_truth_answer": s.ground_truth_answer,
                "ground_truth_sources": s.ground_truth_sources,
                "metadata": s.metadata,
            }
            for s in self.samples
        ]
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
