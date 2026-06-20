# Engineering and Deployment

This document describes how the system is built and served, with a focus on the
model-serving layer. The serving layer runs as a single Hugging Face Inference
Endpoint that loads and orchestrates three models.

## Serving design

The system is served through a Hugging Face Inference Endpoint implemented as an
`EndpointHandler` class (`src/handler.py`). A single endpoint loads and
coordinates the full pipeline rather than running three separate services:

1. The RARE generator (Llama-3.1-8B with LoRA adapter)
2. The error detector (SciBERT sequence classifier)
3. The self-corrector (Flan-T5)

On initialization, the handler authenticates to Hugging Face using a token read
from the environment (`HF_TOKEN`), then loads each model. Hosting all three
models in one endpoint avoids network hops between services during a single
question's generate, detect, and correct cycle.

## Loading the quantized generator

Serving an 8B model on constrained hardware required several specific choices:

- 4-bit quantization (QLoRA, NF4): the base Llama-3.1-8B is loaded in 4-bit
  precision using bitsandbytes so it fits in available GPU memory.
- LoRA adapter loading (PEFT): the fine-tuned LoRA adapter (rank 8, alpha 16) is
  applied on top of the quantized base model.
- Vocabulary resize before adapter load: the model uses two custom reasoning
  tokens (`[REASONING]` and `[/REASONING]`), expanding the vocabulary from
  128,256 to 128,258. Token embeddings are resized to the new vocabulary size
  before the LoRA adapter is loaded; without this step, loading fails with a size
  mismatch.
- CUDA backend preference: the handler sets the preferred linear-algebra backend
  to cusolver to avoid a class of CUDA errors during inference.

## Runtime asset loading

Retrieval assets are not bundled into the image. They are downloaded at runtime
from the Hugging Face Hub using `hf_hub_download`:

- the FAISS index over the medical corpus, and
- the document store used for retrieval.

This keeps the deployment artifact small and allows the index to be updated
independently of the serving code.

## Production thresholds

Pipeline behavior is governed by explicit thresholds set in the handler:

| Threshold | Value | Purpose |
|---|---|---|
| Error detection | 0.5 | Above this, an answer is treated as likely-erroneous |
| Correction | 0.6 | Confidence required to apply a correction |
| Uncertainty | 0.3 | Below this, the answer is flagged as uncertain |
| Minimum answer length | 5 | Reject degenerate, too-short outputs |
| Maximum correction ratio | 2.0 | Prevent corrections from inflating answer length |

Defining these as named thresholds keeps the system's behavior adjustable and
auditable.

## Robustness

The handler includes defensive handling for production reliability:

- Tokenization fallback: if tokenization with the custom reasoning tokens fails,
  the handler falls back to tokenizing without them.
- Bounded generation: generation length is capped for both the generator and the
  Flan-T5 corrector to control latency and cost.
- Graceful degradation: the frontend (`src/app.py`) continues to function with
  caching disabled if the secure Redis cache is unavailable.

## Application and compliance layer

- Frontend: Streamlit (`src/app.py`) sends questions to the endpoint and renders
  the answer, reasoning, and status.
- Caching: a secure Redis cache (TLS, authenticated) stores recent question and
  answer pairs, with a clean fallback when it is not reachable.
- HIPAA-aware handling (`src/hipaa_compliance.py`): PHI anonymization before
  processing, and session-level audit logging of actions including session start,
  question submitted, PHI anonymized, and answer generated.

## To confirm before publishing

The Hugging Face model repository IDs for the error detector and corrector, and
the base-model path, are referenced in `src/handler.py`. Verify they point to the
intended public repositories and that none require a token a user would not have.
