from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag_detail_classifier import RAG_DETAIL_LABELS, labels_to_multihot


@dataclass(frozen=True)
class RagDetailExample:
    text: str
    details: tuple[str, ...]


class RagDetailDataset(Dataset):
    def __init__(self, examples: list[RagDetailExample], tokenizer, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(labels_to_multihot(example.details), dtype=torch.float)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune KLUE-BERT for RAG detail classification.")
    parser.add_argument("--data", default="app/data/rag_domain_train.jsonl")
    parser.add_argument("--output-dir", default="models/rag-detail-klue-bert")
    parser.add_argument("--base-model", default="klue/bert-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_examples(Path(args.data))
    train_examples, validation_examples = split_examples(examples, args.validation_split)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(RAG_DETAIL_LABELS),
        id2label={index: label for index, label in enumerate(RAG_DETAIL_LABELS)},
        label2id={label: index for index, label in enumerate(RAG_DETAIL_LABELS)},
        problem_type="multi_label_classification",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(RagDetailDataset(train_examples, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(RagDetailDataset(validation_examples, tokenizer, args.max_length), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(train_examples).to(device))

    best_accuracy = 0.0
    output_dir = Path(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, validation_loader, criterion, device)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, **metrics}, ensure_ascii=False))
        if metrics["exact_match_accuracy"] >= best_accuracy:
            best_accuracy = metrics["exact_match_accuracy"]
            save_model(output_dir, model, tokenizer, metrics)
    print(f"Saved best model to {output_dir}")


def load_examples(path: Path) -> list[RagDetailExample]:
    resolved_path = path
    if not resolved_path.exists() and not resolved_path.is_absolute():
        backend_relative_path = Path(__file__).resolve().parents[1] / resolved_path
        if backend_relative_path.exists():
            resolved_path = backend_relative_path

    examples: list[RagDetailExample] = []
    labels = set(RAG_DETAIL_LABELS)
    for line_number, line in enumerate(resolved_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "")).strip()
        raw_details = payload.get("expected_details") or [payload.get("rag_detail", "unknown")]
        details = tuple(dict.fromkeys(str(detail).strip() or "unknown" for detail in raw_details))
        if not text:
            raise ValueError(f"Missing text at {resolved_path}:{line_number}")
        unknown_details = [detail for detail in details if detail not in labels]
        if unknown_details:
            raise ValueError(f"Unsupported detail {unknown_details!r} at {resolved_path}:{line_number}")
        examples.append(RagDetailExample(text=text, details=details))
    if not examples:
        raise ValueError(f"No examples found in {resolved_path}")
    return examples


def split_examples(examples: list[RagDetailExample], validation_split: float) -> tuple[list[RagDetailExample], list[RagDetailExample]]:
    shuffled = list(examples)
    random.shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_split))
    return shuffled[validation_count:], shuffled[:validation_count]


def class_pos_weight(examples: list[RagDetailExample]) -> torch.Tensor:
    label_counts = torch.zeros(len(RAG_DETAIL_LABELS), dtype=torch.float)
    for example in examples:
        label_counts += torch.tensor(labels_to_multihot(example.details), dtype=torch.float)
    negative_counts = max(len(examples), 1) - label_counts
    return (negative_counts / label_counts.clamp(min=1.0)).clamp(min=1.0, max=12.0)


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        output = model(**batch)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader: DataLoader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch)
            total_loss += criterion(output.logits, labels).item()
            predictions = torch.sigmoid(output.logits) >= 0.45
            expected = labels >= 0.5
            correct += (predictions == expected).all(dim=-1).sum().item()
            total += labels.shape[0]
    return {"validation_loss": total_loss / max(len(loader), 1), "exact_match_accuracy": correct / max(total, 1)}


def save_model(output_dir: Path, model, tokenizer, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
