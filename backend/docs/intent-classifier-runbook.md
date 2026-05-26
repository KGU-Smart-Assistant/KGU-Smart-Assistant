# Intent Classifier Runbook

This backend can route chat requests with a fine-tuned KLUE-BERT classifier before falling back to Gemini.

## Decision Shape

The KLUE-BERT classifier is responsible for the top-level route and can also
carry the relational DB subtype when the label includes it:

- `route`: `llm`, `relational_db`, `rag`, `weather`
- optional `db_intent`: `map`, `phone`, `unknown`

When `route` is `relational_db`, the backend uses the classifier-provided
`db_intent`. This keeps the chat path rule-base-free; uncertain relational DB
queries should use `relational_db` with `db_intent=unknown` and ask the user to
clarify rather than guessing map vs phone from keywords.

The model should use these labels:

- `llm`
- `relational_db`
- `relational_db:map`
- `relational_db:phone`
- `rag`
- `weather`

For backward compatibility, the backend still accepts legacy labels such as
`map` and `phone`; it normalizes them to `route=relational_db` with the matching
`db_intent`.

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

The current seed builder keeps `db_intent` metadata in the JSONL file so
training can preserve `relational_db:map` and `relational_db:phone` labels.

- `rag`: 184
- `relational_db`: unknown DB examples
- `relational_db:map`: campus location/path examples
- `relational_db:phone`: campus phone/contact examples
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
INTENT_CLASSIFIER_MODEL_NAME=models/intent-klue-bert-v5
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

The backend uses the KLUE-BERT classifier when `INTENT_CLASSIFIER_MODEL_NAME`
is configured and the prediction confidence is at least
`INTENT_CLASSIFIER_CONFIDENCE_THRESHOLD`.

When KLUE-BERT is unavailable or below threshold, routing falls back to the LLM
planner. Keyword-based route guardrails are not used. Compound questions are
split only by the LLM planner response.

RAG domain classification is a separate optional model. It runs only after the
top-level route is `rag` and predicts multiple domain labels with sigmoid
scores. RAG domain classification intentionally does not fall back to keyword
rules; when this model is not configured or returns no accepted labels, the
domain is reported as `unknown`.

After a query is routed to RAG, the orchestrator now treats intent enrichment as
a six-stage pipeline:

1. `routing`: choose `llm`, `relational_db`, `rag`, or `weather` with KLUE-BERT
   or the LLM planner fallback.
2. `rag_domain`: use the dedicated multi-label KLUE-BERT domain classifier.
3. `rag_detail`: classify the detail axis such as `period`, `eligibility`, or
   `required_documents` with the optional RAG detail KLUE-BERT model. This
   stage intentionally does not fall back to keyword rules; when the model is
   not configured or returns a low-confidence detail, the detail is `unknown`.
4. `confidence`: calculate a confidence score from domain score shape and the
   detail model signal.
5. `ambiguity`: report `clear`, `multi_domain`, `low_confidence`,
   `missing_detail`, or `needs_clarification`.
6. `rewritten_queries`: expose retrieval-ready query variants for downstream
   search improvements.

The API response includes `rag_ambiguity` and `rewritten_queries` alongside the
existing `rag_domain`, `rag_domains`, `rag_detail`, and `rag_confidence` fields.
When the answer is blocked for clarification, it also includes
`suggested_domains` and `suggested_details` so the frontend can render guided
choice buttons instead of treating the response as a generic failure.
The current domain stage is model-only; the remaining stages are separated so
they can be replaced by trained models without changing the response contract.

Configure it separately:

```env
RAG_DOMAIN_CLASSIFIER_MODEL_NAME=models/rag-domain-klue-bert-v3
RAG_DOMAIN_CLASSIFIER_CONFIDENCE_THRESHOLD=0.5
RAG_DOMAIN_CLASSIFIER_TOP_K=3
RAG_DOMAIN_CLASSIFIER_DEVICE=-1
RAG_DETAIL_CLASSIFIER_MODEL_NAME=models/rag-detail-klue-bert-v6
RAG_DETAIL_CLASSIFIER_CONFIDENCE_THRESHOLD=0.45
RAG_DETAIL_CLASSIFIER_TOP_K=3
RAG_DETAIL_CLASSIFIER_DEVICE=-1
RAG_CLARIFY_ON_LOW_CONFIDENCE=true
RAG_CLARIFY_ON_MULTI_DOMAIN=true
RAG_CLARIFY_ON_MISSING_DETAIL=false
```

`RAG_CLARIFY_ON_LOW_CONFIDENCE` and `RAG_CLARIFY_ON_MULTI_DOMAIN` should stay
enabled in production. They prevent the bot from searching and generating an
answer when the model-only RAG intent is uncertain. `missing_detail` is allowed
by default because many domain-only questions can still be answered from the
retrieved documents; enable `RAG_CLARIFY_ON_MISSING_DETAIL` only if the frontend
wants a stricter guided-chat flow.

Train it from the RAG intent evaluation data:

```bash
python scripts/train_rag_domain_classifier.py \
  --data app/data/rag_domain_train.jsonl \
  --output-dir models/rag-domain-klue-bert-v2 \
  --epochs 3
```

Regenerate the expanded training data before retraining:

```bash
python scripts/build_rag_domain_training_data.py
```

Evaluate it on the held-out domain test set:

```bash
python scripts/evaluate_rag_domain_classifier.py \
  --data app/data/rag_domain_test.jsonl \
  --model models/rag-domain-klue-bert-v2 \
  --threshold 0.5 \
  --pretty
```

Train the optional detail classifier from the same expanded data. The detail
axis is multi-label, so rows can use `expected_details` when one question spans
multiple details:

```bash
python scripts/train_rag_detail_classifier.py \
  --data app/data/rag_detail_train.jsonl \
  --output-dir models/rag-detail-klue-bert-v5 \
  --epochs 3
```

Evaluate the detail model separately:

```bash
python scripts/evaluate_rag_detail_classifier.py \
  --data app/data/rag_detail_test.jsonl \
  --model models/rag-detail-klue-bert-v5 \
  --threshold 0.45 \
  --top-k 3 \
  --pretty
```

## Validate One Input

Use the validation script to check how the trained model classifies a specific
question before running the full backend.

```bash
python scripts/validate_intent_classifier.py \
  --model models/intent-klue-bert \
  --text "성적향상 장학금은 어디에서 정보를 찾을 수 있어?" \
  --expected-route rag \
  --pretty
```

Example relational DB route check:

```bash
python scripts/validate_intent_classifier.py \
  --model models/intent-klue-bert \
  --text "8강의동은 어디야?" \
  --expected-route relational_db \
  --pretty
```

The script exits with code `1` when `--expected-route` does not match the model
prediction, so it can also be used in quick local regression checks. Use the
chat planner validation script when you specifically need to verify
`relational_db:map` or `relational_db:phone` resolver behavior.

## Validate The Chat Planner

Use the planner validation script to check how one user question is decomposed
into executable actions. This does not require the KLUE-BERT model unless your
environment enables `INTENT_CLASSIFIER_MODEL_NAME`.

```bash
python scripts/validate_chat_planner.py \
  --text "학생회관 위치랑 전화번호 알려줘" \
  --expected-action relational_db:map \
  --expected-action relational_db:phone \
  --pretty
```

For questions that mix document search and live services:

```bash
python scripts/validate_chat_planner.py \
  --text "장학금 신청 기간하고 내일 수원 날씨 알려줘" \
  --expected-action rag \
  --expected-action weather \
  --pretty
```

The script exits with code `1` when the planned actions do not match the
expected actions. Inspect the `query` field in the output to confirm that each
downstream service receives the right atomic question.

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

## Add Production Failure Cases

Add real failed questions to `app/data/intent_failure_cases.jsonl`. This is the
single hand-maintained file for production-like intent errors. Each row should
include:

- `text`: the exact user question.
- `expected_route`: `llm`, `relational_db`, `rag`, or `weather`.
- `expected_db_intent`: `map`, `phone`, or `unknown`.
- `expected_rag_domain`, `expected_rag_domains`, `expected_rag_detail`, and
  `expected_ambiguity` for RAG rows.
- `failure_type`: a short bucket such as `where_rag_not_map`, `multi_domain`,
  `short_detail`, `procedure`, or `phone_contact`.
- `memo`: why the case matters.

The training data builders automatically include these rows:

```bash
python scripts/build_intent_training_seed.py
python scripts/build_rag_domain_training_data.py
python scripts/build_rag_detail_training_data.py
```

Before retraining, inspect the collected production-like cases:

```bash
python scripts/evaluate_intent_failure_cases.py \
  --data app/data/intent_failure_cases.jsonl \
  --pretty \
  --show-errors
```
