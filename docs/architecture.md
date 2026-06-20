# Architecture

RARE is a modular pipeline. It grounds answers in retrieved evidence, then uses
an independent model to verify each answer and either correct it or pass it
through. Generation, verification, and correction are handled by separate models
so that the system can detect its own errors.

## Architecture flow

![RARE architecture flow diagram](images/architecture.png)

A question enters the RARE generator, which produces an evidence-grounded answer.
The error detector then classifies that answer:

- No error: the answer becomes the final answer.
- Error detected: the answer is sent to the error corrector, which produces a
  corrected response. The corrected input can be re-run through the RARE
  generator (the feedback loop), and the corrected response is returned as the
  final answer.

## Components

| Component | Model | Role |
|---|---|---|
| Retriever | SentenceBERT (MiniLM) with FAISS | Finds relevant medical passages |
| RARE generator | Llama-3.1-8B with LoRA (4-bit QLoRA), reasoning tokens | Generates grounded answers |
| Error detector | SciBERT | Binary classification: likely correct or likely error |
| Self-corrector | Flan-T5 | Rewrites flagged answers using the evidence |

### Retriever

SentenceBERT (MiniLM) builds a dense vector index over a deduplicated medical
corpus and returns the top-k passages for each question. Retrieval runs at both
training and inference time, so retrieved evidence shapes reasoning rather than
acting as a late-stage addition.

### RARE generator

Llama-3.1-8B fine-tuned with adapter-based LoRA (rank 8, alpha 16) under 4-bit
NF4 quantization (bitsandbytes), allowing an 8B model to train and serve on
limited hardware. Custom `[REASONING]` and `[/REASONING]` tokens separate the
model's reasoning from its final answer; the vocabulary is resized to accommodate
the added tokens. Evidence is injected into training prompts, following the RARE
approach of separating knowledge memorization from reasoning optimization.

### Error detector

A SciBERT binary classifier trained on biomedical text, used to judge whether a
generated answer is likely incorrect. Its training data was generated
automatically by scoring RARE outputs against reference answers, with low overlap
labeled as an error.

### Self-corrector

A Flan-T5 model that produces a corrected version of a flagged answer,
conditioned on the question and evidence, while preserving the answer's intent.
It is trained on the error cases only.

## Application layer

The models are served through a Hugging Face Inference Endpoint
(`src/handler.py`) and consumed by a Streamlit frontend (`src/app.py`). A
compliance module (`src/hipaa_compliance.py`) provides PHI anonymization before
processing, session-level audit logging, and a secure Redis cache that the
application operates without when it is unavailable. Serving detail is in
`docs/engineering.md`.
