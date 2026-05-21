from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_MODEL_DIR = "models/intent-klue-bert"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the trained KLUE-BERT intent classifier to Hugging Face Hub."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--repo-id", required=True, help="Example: KGU-Smart-Assistant/kgu-intent-klue-bert")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Upload KLUE-BERT intent classifier",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    validate_model_dir(model_dir)

    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi()
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=args.commit_message,
    )
    print(f"Published {model_dir} to https://huggingface.co/{args.repo_id}")


def validate_model_dir(model_dir: Path) -> None:
    required_files = ("config.json", "tokenizer_config.json")
    missing_files = [name for name in required_files if not (model_dir / name).exists()]
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))

    if missing_files:
        raise FileNotFoundError(f"Missing model files in {model_dir}: {', '.join(missing_files)}")
    if not has_weights:
        raise FileNotFoundError(f"Missing model weights in {model_dir}")


if __name__ == "__main__":
    main()
