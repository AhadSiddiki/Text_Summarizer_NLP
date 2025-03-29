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
```
# Text from  sample.txt
SpaceX’s Starship Catcher, often referred to as the “Mechazilla” system, is an ambitious and innovative approach to rocket recovery and reusability. This catching system is designed to retrieve the Super Heavy booster—the first stage of the Starship launch system—using large mechanical arms attached to the launch tower. The concept is meant to streamline recovery operations, reduce turnaround time, and significantly lower the cost of space travel.

How the Catcher Works
Instead of relying on traditional landing legs, the Super Heavy booster will descend toward the launch tower after launch, using its grid fins for control. The massive robotic arms, which resemble giant “chopsticks,” will then position themselves to catch the descending booster just above the ground. The goal is to clamp onto the booster around the grid fins and securely hold it in place before lowering it onto the launch pad for reuse.

By eliminating landing legs, SpaceX aims to reduce the booster’s weight, increasing payload capacity and simplifying the overall design. This method also minimizes wear and tear that would occur from landing on hard surfaces, potentially extending the lifespan of each booster.

Challenges and Risks
While the concept is revolutionary, it comes with significant engineering and operational challenges. The precision required for the booster to align with the catcher arms is extreme. Any deviation in trajectory or timing could result in a catastrophic failure. Moreover, the mechanical arms must be robust enough to withstand the impact forces while maintaining a high degree of flexibility and control.

Weather conditions, such as strong winds or turbulence, could also pose a challenge during the catching process. SpaceX engineers are actively refining guidance systems, real-time adjustments, and backup safety protocols to mitigate these risks.

Starship and Future Applications
The Starship system, which includes the booster and the upper-stage spacecraft, is designed for deep-space missions, including trips to the Moon and Mars. If the catcher system proves successful, it could enable a rapid launch turnaround—possibly within hours—making Starship the first fully reusable rocket system capable of frequent space missions.

In the long term, this technology could revolutionize space travel by making it as routine as air travel. By cutting launch costs and enabling high-frequency launches, SpaceX’s Starship Catcher could play a pivotal role in the company’s vision of making life multiplanetary.
```
```bash
python txt_summarize.py --input sample.txt --output summary.txt --method abstractive --model google/pegasus-xsum --min_length 100
```
[View Summary Text File](summary.txt)
```
# After Summarize summary.txt
SpaceX, the private rocket company, is developing a robotic system to recover and reuse its Super Heavy rocket booster after a launch, which could radically change the way space missions are launched and reusability is achieved, according to the company’s founder and chief executive officer, Elon Musk, who unveiled the concept at the International Space Station (ISS) conference in March.
 If the catcher system proves successful, it could enable a rapid launch turnaround—possibly within hours—making Starship the first fully reusable rocket system capable of frequent space launches.
```


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
