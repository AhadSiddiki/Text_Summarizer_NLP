# Text Summarizer NLP

A powerful text summarization tool implemented using state-of-the-art NLP techniques. This tool supports both extractive, abstractive, and hybrid summarization approaches.

## Features

- **Extractive Summarization**: Selects the most important sentences from the original text
- **Abstractive Summarization**: Generates new sentences that capture the essence of the original text
- **Hybrid Summarization**: Combines both approaches for optimal results
- **Multiple Model Options**: Choose from various state-of-the-art transformer models
- **Evaluation Metrics**: Built-in ROUGE score calculation
- **Multiple Input Methods**: Accepts text directly or from files
- **Command-line Interface**: Easy to use from the terminal

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the spaCy model:

```bash
python -m spacy download en_core_web_md
```

## Usage

### Command Line Interface

```bash
python txt_summarize.py --input sample.txt --output summary.txt --method hybrid
```

#### Arguments

- `--input`: Input text or file path (optional, can enter text interactively)
- `--output`: Output file path for summary (optional, prints to console if not specified)
- `--method`: Summarization method: "extractive", "abstractive", or "hybrid" (default: "hybrid")
- `--model`: Hugging Face model name for summarization (default: "google/pegasus-large")
- `--max_length`: Maximum summary length (default: 150)
- `--min_length`: Minimum summary length (default: 40)
- `--sentences`: Number of sentences for extractive summarization (default: 5)
- `--extractive_method`: Method for extractive summarization: "tfidf" or "textrank" (default: "textrank")
- `--extractive_ratio`: Ratio for extractive step in hybrid summarization (default: 0.5)

### Available Models

You can specify different models using the `--model` parameter:

- `google/pegasus-large` (default): Google's Pegasus model for more creative, higher-quality summaries
- `google/pegasus-xsum`: Optimized for extreme summarization (very concise summaries)
- `facebook/bart-large-cnn`: Facebook's BART model fine-tuned on CNN articles
- `t5-large`: Google's T5 large model for diverse summarization
- `allenai/led-large-16384`: Longformer model for handling very long documents

Example:

[View Sample Text File](sample.txt)

```bash
python txt_summarize.py --input sample.txt --output summary.txt --method abstractive --model google/pegasus-xsum --min_length 100
```
[View Summary Text File](summary.txt)



### Python API

```python
from txt_summarize import TextSummarizer

# Initialize the summarizer with a specific model
summarizer = TextSummarizer(model_name="google/pegasus-large")

# Abstractive summarization
abstract_summary = summarizer.abstractive_summarize(
    "Your long text here...",
    max_length=150,
    min_length=40
)

# Extractive summarization
extract_summary = summarizer.extractive_summarize(
    "Your long text here...",
    num_sentences=5,
    method="textrank"  # or "tfidf"
)

# Hybrid summarization (recommended)
hybrid_summary = summarizer.hybrid_summarize(
    "Your long text here...",
    extractive_ratio=0.5,
    max_length=150,
    min_length=40,
    extractive_method="textrank"
)

# Evaluate a summary
scores = summarizer.evaluate_summary("Original text...", hybrid_summary)
print(f"ROUGE-1: {scores['rouge1']:.4f}")
```

## Evaluation

The summarizer calculates ROUGE scores to evaluate summary quality:

- **ROUGE-1**: Overlap of unigrams
- **ROUGE-2**: Overlap of bigrams
- **ROUGE-L**: Longest Common Subsequence

## Contributing

Contributions to this project are welcome! Feel free to submit issues, feature requests, or pull requests to help improve this image caption generator.

## Contact

- GitHub: [https://github.com/AhadSiddiki](https://github.com/AhadSiddiki)
- Email: [ahad.siddiki25@gmail.com](mailto:ahad.siddiki25@gmail.com)
- LinkedIn: [http://www.linkedin.com/in/ahad-siddiki/](http://www.linkedin.com/in/ahad-siddiki/)
- Instagram: [www.instagram.com/ahad.siddiki/](https://www.instagram.com/ahad.siddiki/)
