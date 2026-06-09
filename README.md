# image2mcq

Convert images — screenshots, scanned pages, diagrams, charts, and photographs — into high-quality MCQ questions using AI.

Built on top of **html2mcq**'s image pipeline, extracted as a standalone library focused purely on image-to-MCQ generation.

---

## Features

- **Two processing methods:**
  - `twostep` (default) — OCR image text, then generate MCQs from extracted text
  - `images2mcq` — send images directly to a vision LLM for MCQ generation
- **Multiple AI providers:** OpenRouter, Anthropic, OpenAI, Ollama
- **Auto model failover:** if one model fails (e.g. quota exhausted), automatically tries the next
- **Local OCR fallback:** Tesseract OCR when vision APIs are unavailable
- **CLI & Python API** — use from terminal or integrate into your code

---

## Quick Start

### CLI

```bash
# Single image file
image2mcq --image-path diagram.png -n 5

# Multiple image URLs
image2mcq --image-url https://example.com/chart1.png --image-url https://example.com/chart2.png

# Scan a folder of images
image2mcq --image-folder ./lecture-slides/ --method images2mcq

# Output as JSON
image2mcq --image-path notes.png -o questions.json --format json

# Use n=999 to generate as many as the content supports
image2mcq --image-path textbook-page.png

# Override OCR model per call
image2mcq --image-folder ./slides/ --ocr-model "google/gemini-2.5-flash-lite"

# Override MCQ model per call
image2mcq --image-path diagram.png --mcq-model "openai/gpt-4o"

# Save OCR text to file
image2mcq --image-path scanned-page.png --method twostep --save-ocr-path ocr_output.txt

# Custom instructions
image2mcq --image-path graph.png -i "Make answers very close and confusing"

# Show progress bar during MCQ generation
image2mcq --image-folder ./slides/ --progress

# Local Ollama
image2mcq --image-path diagram.png --provider ollama --mcq-model qwen2.5:7b
```

### Python API

```python
from image2mcq import ImageMCQGenerator

gen = ImageMCQGenerator(
    api_key="sk-or-v1-...",
    provider="openrouter",
    mcq_model="google/gemini-2.5-flash-lite",
)

# From local files
mcq = gen.from_image_paths("screenshot.png", n=5)
print(mcq.to_pretty_str())

# From URLs
mcq = gen.from_image_urls("https://example.com/diagram.png", n=3)
print(mcq.to_json())

# From multiple images
mcq = gen.from_image_paths(["page1.png", "page2.png", "page3.png"])

# Show progress bar during generation
mcq = gen.from_image_paths("lecture-slide.png", n=5, show_progress=True)
```

### Two-Step (OCR → MCQ)

```python
gen = ImageMCQGenerator(
    api_key="sk-or-v1-...",
    method="twostep",  # default
    ocr_model="google/gemini-2.5-flash-lite",
    save_ocr_path="ocr_output.txt",  # save OCR text to file
)
mcq = gen.from_image_paths("scanned-page.png", n=10)
```

### Images2MCQ (Vision Direct)

```python
gen = ImageMCQGenerator(
    api_key="sk-or-v1-...",
    method="images2mcq",
    mcq_model="openai/gpt-4o",  # vision model
)
mcq = gen.from_image_paths("architecture-diagram.png", n=5)
```

### Custom Instructions & Overrides

```python
mcq = gen.from_image_paths(
    "graph.png",
    n=5,
    difficulty_mix="50% easy, 50% hard",
    focus_topics=["data structures", "time complexity"],
    custom_instructions="Make answers very close and confusing",
    ocr_model="google/gemini-2.5-flash-lite",  # per-call override
    mcq_model="openai/gpt-4o",                  # per-call override
    api_key_override="sk-or-v1-...",            # different key for this call
    prompt_log_path="stdout",                   # log prompts for this call
)
```

### Auto Model Selection

```python
gen = ImageMCQGenerator(
    api_key="sk-or-v1-...",
    mcq_model="auto",
    mcq_model_list=[
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "google/gemma-4-31b-it:free",
    ],
)
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Default API key for OpenRouter |
| `ANTHROPIC_API_KEY` | API key for Anthropic |
| `OPENAI_API_KEY` | API key for OpenAI |
| `IMAGE2MCQ_MCQ_MODELS` | Comma-separated MCQ model priority list for `model="auto"` |
| `IMAGE2MCQ_OCR_MODELS` | Comma-separated OCR model priority list for `ocr_model="auto"` |

---

## Output Format

```python
# Pretty-print
print(mcq.to_pretty_str())

# JSON
print(mcq.to_json())
# {
#   "total_exam_time": 20,
#   "questions": [
#     {
#       "question_html": "What is the time complexity of binary search?",
#       "options": ["O(n)", "O(log n)", "O(n^2)", "O(1)"],
#       "answers": [1],
#       "multi": false,
#       "marks": 1.0,
#       "negative_marks": 0.25,
#       "difficulty": "easy",
#       "explanation": "Binary search halves the search space each iteration."
#     }
#   ]
# }
```

---

## Installation

```bash
pip install image2mcq
```

For OCR support, also install [Tesseract](https://github.com/tesseract-ocr/tesseract).

---

## License

MIT
