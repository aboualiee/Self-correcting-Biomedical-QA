# RARE: Self-Correcting Retrieval-Augmented Reasoning for Biomedical QA

RARE is a biomedical question-answering system that combines Retrieval-Augmented
Reasoning with a self-correction pipeline. It generates answers grounded in
retrieved medical literature, uses an independent model to detect likely errors,
applies a corrector to flagged answers, and serves the result through a
HIPAA-aware application layer.

This is a research prototype for biomedical QA. It is not a medical device and is
not intended for clinical use. See the [model card](docs/model_card.md) for
intended use and limitations.

## Documentation

- [Architecture](docs/architecture.md): system design and data flow, with the
  pipeline diagram.
- [Engineering and deployment](docs/engineering.md): how the three models are
  loaded, quantized, and served behind a single Hugging Face endpoint.
- [Model card](docs/model_card.md): intended use, training data, evaluation
  results, and limitations.

## Pipeline

1. Retrieve relevant passages from a medical corpus (SentenceBERT with FAISS).
2. Generate an answer with a fine-tuned Llama-3.1-8B that injects explicit
   reasoning tokens.
3. Detect likely errors in the generated answer with a SciBERT classifier.
4. Correct flagged answers with a Flan-T5 model.
5. Serve through a HIPAA-aware layer (PHI anonymization, audit logging, secure
   caching).

## Components

| Component | Model | Role |
|---|---|---|
| Retriever | SentenceBERT (MiniLM) with FAISS | Finds relevant medical passages |
| RARE generator | Llama-3.1-8B with LoRA (4-bit QLoRA), reasoning tokens | Generates grounded answers |
| Error detector | SciBERT | Binary classification: likely correct or likely error |
| Self-corrector | Flan-T5 | Rewrites flagged answers using the evidence |

## Repository layout

```
.
├── src/
│   ├── app.py               Streamlit frontend
│   ├── handler.py           Hugging Face Inference Endpoint handler (runs the pipeline)
│   └── hipaa_compliance.py  PHI anonymization, audit logging, secure Redis
├── tests/
│   └── auth_test.py         Hugging Face token and access checks
├── docs/
│   ├── architecture.md      System design and data flow
│   ├── engineering.md       Model serving and deployment detail
│   ├── model_card.md        Intended use, data, metrics, limitations
│   └── images/              Diagrams and result charts
├── requirements.txt
├── .env.example
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # set your own values
streamlit run src/app.py
```

## Configuration

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token for model access |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_USERNAME`, `REDIS_PASSWORD` | Secure cache (optional; the app runs without it) |

Do not commit `.env`. A template is provided in [`.env.example`](.env.example).

## Evaluation

The system is evaluated per component using ROUGE-L, BERTScore, BLEU, and
classification accuracy and F1. Full results, including the metrics chart and
known limitations, are in the [model card](docs/model_card.md).

## Resources

- [Research Poster](https://drive.google.com/file/d/1Eq3vYundTcvwx023KoGUk91V1GdpHMe_/view)
- [Demo Video](https://youtu.be/2kyPwGk86YU)
- [Datasets on Hugging Face](https://huggingface.co/datasets/Maikobi/RARE_output_and_generated_datasets)

## Origin

Developed as a master's thesis project at EPITA Graduate School of Computer
Science (2025). The baseline RARE generator was implemented by Abubakar Aliyu.
The full self-correcting framework was a four-person team effort covering error
detection, correction, and retrieval, advised by Alaa Bakhti.

## License

Released under the MIT License. See [LICENSE](LICENSE).