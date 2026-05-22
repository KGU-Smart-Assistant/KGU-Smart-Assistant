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

from app.services.rag_domain_classifier import RAG_DOMAIN_LABELS, labels_to_multihot


@dataclass(frozen=True)
class RagDomainExample:
    text: str
    domains: tuple[str, ...]


class RagDomainDataset(Dataset):
    def __init__(self, examples: list[RagDomainExample], tokenizer, max_length: int) -> None:
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
        item["labels"] = torch.tensor(labels_to_multihot(example.domains), dtype=torch.float)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune KLUE-BERT for multi-label RAG domain classification.")
    parser.add_argument("--data", default="app/data/rag_intent_eval.jsonl")
    parser.add_argument("--output-dir", default="models/rag-domain-klue-bert")
    parser.add_argument("--base-model", default="klue/bert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_examples(Path(args.data))
    train_examples, validation_examples = split_examples(
        examples=examples,
        validation_split=args.validation_split,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(RAG_DOMAIN_LABELS),
        id2label={index: label for index, label in enumerate(RAG_DOMAIN_LABELS)},
        label2id={label: index for index, label in enumerate(RAG_DOMAIN_LABELS)},
        problem_type="multi_label_classification",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        RagDomainDataset(train_examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        RagDomainDataset(validation_examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_pos_weight(train_examples).to(device))
    best_f1 = 0.0
    output_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device, threshold=args.threshold)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **validation_metrics,
                },
                ensure_ascii=False,
            )
        )

        if validation_metrics["micro_f1"] >= best_f1:
            best_f1 = validation_metrics["micro_f1"]
            save_model(output_dir, model, tokenizer, validation_metrics)

    print(f"Saved best model to {output_dir}")


def load_examples(path: Path) -> list[RagDomainExample]:
    resolved_path = path
    if not resolved_path.exists() and not resolved_path.is_absolute():
        backend_relative_path = Path(__file__).resolve().parents[1] / resolved_path
        if backend_relative_path.exists():
            resolved_path = backend_relative_path

    examples: list[RagDomainExample] = []
    labels = set(RAG_DOMAIN_LABELS)
    for line_number, line in enumerate(resolved_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "")).strip()
        raw_domains = payload.get("expected_domains") or [payload.get("rag_domain") or payload.get("domain")]
        domains = tuple(dict.fromkeys(str(domain).strip() for domain in raw_domains if str(domain).strip()))
        unknown_domains = [domain for domain in domains if domain not in labels]
        if not text:
            raise ValueError(f"Missing text at {resolved_path}:{line_number}")
        if unknown_domains:
            raise ValueError(f"Unsupported domains {unknown_domains!r} at {resolved_path}:{line_number}")
        examples.append(RagDomainExample(text=text, domains=domains))

    if not examples:
        raise ValueError(f"No examples found in {resolved_path}")
    return examples


def split_examples(
    examples: list[RagDomainExample],
    validation_split: float,
) -> tuple[list[RagDomainExample], list[RagDomainExample]]:
    shuffled = list(examples)
    random.shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_split))
    return shuffled[validation_count:], shuffled[:validation_count]


def class_pos_weight(examples: list[RagDomainExample]) -> torch.Tensor:
    label_counts = torch.zeros(len(RAG_DOMAIN_LABELS), dtype=torch.float)
    for example in examples:
        label_counts += torch.tensor(labels_to_multihot(example.domains), dtype=torch.float)
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


def evaluate(model, loader: DataLoader, criterion, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            output = model(**batch)
            total_loss += criterion(output.logits, labels).item()
            probabilities = torch.sigmoid(output.logits)
            predictions = probabilities >= threshold
            expected = labels >= 0.5
            true_positive += (predictions & expected).sum().item()
            false_positive += (predictions & ~expected).sum().item()
            false_negative += (~predictions & expected).sum().item()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    micro_f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "validation_loss": total_loss / max(len(loader), 1),
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
    }


def save_model(output_dir: Path, model, tokenizer, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
