# Intent Classifier Runbook

This backend can route chat requests with a fine-tuned KLUE-BERT classifier before falling back to Gemini.

## Decision Shape

Public routing decisions use two fields:

- `route`: `llm`, `relational_db`, `rag`, `weather`
- `db_intent`: `map`, `phone`, `unknown`

When `route` is not `relational_db`, `db_intent` should be `unknown`.

The model may use internal labels such as:

- `llm`
- `rag`
- `weather`
- `relational_db:map`
- `relational_db:phone`
- `relational_db:unknown`

The backend maps those labels back into the public decision shape.

## Train Locally

Install optional ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Rebuild deterministic seed data:

```bash
python scripts/build_intent_training_seed.py
```

Train the model:

```bash
python scripts/train_intent_classifier.py \
  --data app/data/intent_training_seed.jsonl \
  --output-dir models/intent-klue-bert \
  --epochs 5
```

The output directory is ignored by Git because model weights are too large for normal repository history.

The current seed builder creates 818 examples:

- `rag`: 184
- `relational_db:map`: 190
- `relational_db:phone`: 154
- `relational_db:unknown`: 90
- `weather`: 100
- `llm`: 100

## Publish For Production

Create a Hugging Face token with write access and expose it as `HF_TOKEN`.

```bash
python scripts/publish_intent_classifier.py \
  --model-dir models/intent-klue-bert \
  --repo-id dabinyou/kgu-intent-klue-bert
```

Use `--private` only if the team has decided the model should require authenticated access.

## Configure Backend

For local use with a locally trained model:

```env
INTENT_CLASSIFIER_MODEL_NAME=models/intent-klue-bert
INTENT_CLASSIFIER_CONFIDENCE_THRESHOLD=0.7
INTENT_CLASSIFIER_DEVICE=-1
```

For production use with Hugging Face Hub:

```env
INTENT_CLASSIFIER_MODEL_NAME=dabinyou/kgu-intent-klue-bert
INTENT_CLASSIFIER_CONFIDENCE_THRESHOLD=0.7
INTENT_CLASSIFIER_DEVICE=-1
```

If the model repository is private, the runtime environment also needs `HF_TOKEN`.

## Runtime Behavior

The backend uses the KLUE-BERT classifier only when `INTENT_CLASSIFIER_MODEL_NAME` is configured and the prediction confidence is at least `INTENT_CLASSIFIER_CONFIDENCE_THRESHOLD`.

Otherwise it falls back to the existing LLM-based routing flow.

## Validate One Input

Use the validation script to check how the trained model classifies a specific
question before running the full backend.

```bash
python scripts/validate_intent_classifier.py \
  --model models/intent-klue-bert \
  --text "성적향상 장학금은 어디에서 정보를 찾을 수 있어?" \
  --expected-route rag \
  --expected-db-intent unknown \
  --pretty
```

Example map intent check:

```bash
python scripts/validate_intent_classifier.py \
  --model models/intent-klue-bert \
  --text "8강의동은 어디야?" \
  --expected-route relational_db \
  --expected-db-intent map \
  --pretty
```

The script exits with code `1` when `--expected-route` or
`--expected-db-intent` does not match the model prediction, so it can also be
used in quick local regression checks.

## Evaluate The Classifier

Run the held-out evaluation set after training:

```bash
python scripts/evaluate_intent_classifier.py \
  --model models/intent-klue-bert \
  --data app/data/intent_eval.jsonl \
  --threshold 0.7 \
  --pretty \
  --show-errors
```

For CI or a release check, add a minimum accuracy gate:

```bash
python scripts/evaluate_intent_classifier.py \
  --model models/intent-klue-bert \
  --data app/data/intent_eval.jsonl \
  --threshold 0.7 \
  --fail-under 0.9
```

The evaluation set intentionally includes ambiguous Korean questions such as
`어디에서 확인해?` for RAG and `어디야?` for campus map lookup. These examples help
catch the common failure mode where every `어디` question is misrouted to map.

Use the report to tune `INTENT_CLASSIFIER_CONFIDENCE_THRESHOLD`:

- Raise it, for example to `0.8`, when wrong predictions are accepted with high confidence.
- Lower it, for example to `0.6`, only when correct predictions fall back to Gemini too often.

Compound questions such as `중앙도서관 위치랑 전화번호 알려줘` are handled by the
backend planner before KLUE-BERT. Keep those out of the single-label classifier
training/evaluation set unless the model architecture is changed to multi-label.
