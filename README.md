# TOON Libraries: Minimal Benchmark

This is a small, objective experiment that compares three TOON libraries (toonify, toons, toon-formatter) for payload size, token counts, and dumps/loads performance. It also validates typed round-trips with Pydantic. It runs locally with no network calls.

## What it measures

- Bytes and token counts for a small RAG-style context payload.
- Bytes and token counts for a small answer payload.
- Dumps/loads timing across toonify, toons, and toon-formatter.
- Typed validation for TOON outputs using Pydantic models.

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run.py
```

## Notes

- Token counts use `tiktoken` with `gpt-5.2` as the reference tokenizer. Change `MODEL_NAME` in `run.py` if you want a different tokenizer.
- Iteration count is controlled by `ITERATIONS` in `run.py`.
- TOON output uses short keys to reflect the common production pattern where key shortening compounds savings.
- Cost estimation uses the `PRICING` table in `run.py` (USD per 1M tokens).
