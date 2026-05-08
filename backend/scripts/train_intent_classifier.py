from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROUTES = {"llm", "relational_db", "rag", "weather"}
DB_INTENTS = {"map", "phone", "unknown"}
LABELS = [
    "llm",
    "rag",
    "weather",
    "relational_db:map",
    "relational_db:phone",
    "relational_db:unknown",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class IntentExample:
    text: str
    route: str
    db_intent: str = "unknown"

    @property
    def label(self) -> str:
        if self.route == "relational_db":
            return f"{self.route}:{self.db_intent}"
        return self.route


class IntentDataset(Dataset):
    def __init__(self, examples: list[IntentExample], tokenizer, max_length: int) -> None:
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
        item["labels"] = torch.tensor(LABEL_TO_ID[example.label], dtype=torch.long)
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune KLUE-BERT for chat intent routing.")
    parser.add_argument("--data", default="app/data/intent_training_seed.jsonl")
    parser.add_argument("--output-dir", default="models/intent-klue-bert")
    parser.add_argument("--base-model", default="klue/bert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--validation-split", type=float, default=0.2)
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
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        IntentDataset(train_examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        IntentDataset(validation_examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = 0.0
    output_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, device)
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

        if validation_metrics["accuracy"] >= best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            save_model(output_dir, model, tokenizer, validation_metrics)

    print(f"Saved best model to {output_dir}")


def load_examples(path: Path) -> list[IntentExample]:
    examples: list[IntentExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text", "")).strip()
        route = str(payload.get("route", payload.get("label", ""))).strip()
        db_intent = str(payload.get("db_intent", "unknown")).strip()
        if route in {"map", "phone"}:
            db_intent = route
            route = "relational_db"
        if not text:
            raise ValueError(f"Missing text at {path}:{line_number}")
        if route not in ROUTES:
            raise ValueError(f"Unsupported route {route!r} at {path}:{line_number}")
        if db_intent not in DB_INTENTS:
            raise ValueError(f"Unsupported db_intent {db_intent!r} at {path}:{line_number}")
        example = IntentExample(text=text, route=route, db_intent=db_intent)
        if example.label not in LABEL_TO_ID:
            raise ValueError(
                f"Unsupported route/db_intent pair {example.label!r} at {path}:{line_number}"
            )
        examples.append(example)

    if len(examples) < len(LABELS):
        raise ValueError("Training data is too small.")
    return examples


def split_examples(
    examples: list[IntentExample],
    validation_split: float,
) -> tuple[list[IntentExample], list[IntentExample]]:
    by_label: dict[str, list[IntentExample]] = {label: [] for label in LABELS}
    for example in examples:
        by_label[example.label].append(example)

    train_examples: list[IntentExample] = []
    validation_examples: list[IntentExample] = []
    for label, label_examples in by_label.items():
        if not label_examples:
            raise ValueError(f"Missing training examples for label {label!r}")

        random.shuffle(label_examples)
        validation_count = max(1, round(len(label_examples) * validation_split))
        validation_examples.extend(label_examples[:validation_count])
        train_examples.extend(label_examples[validation_count:])

    random.shuffle(train_examples)
    random.shuffle(validation_examples)
    return train_examples, validation_examples


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        total_loss += output.loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            predictions = torch.argmax(output.logits, dim=-1)
            total += batch["labels"].size(0)
            correct += (predictions == batch["labels"]).sum().item()
            total_loss += output.loss.item()

    return {
        "validation_loss": total_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
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
