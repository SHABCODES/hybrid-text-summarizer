# hybrid-text-summarizer
# Hybrid Extractive Text Summarization

A research project implementing a hybrid summarization model combining K-Means Clustering, LDA Topic Modeling, and TextRank algorithms.

## Features

- **Hybrid Approach**: Combines three unsupervised algorithms for robust summarization
- **Multi-language Support**: Framework ready for multiple languages
- **Benchmark Evaluation**: Tested on CNN/DailyMail dataset
- **Comprehensive Metrics**: ROUGE-1, ROUGE-2, ROUGE-L evaluation

## Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.2295 |
| ROUGE-2 | 0.0795 |
| ROUGE-L | 0.1424 |

## Installation

```bash
git clone https://github.com/yourusername/hybrid-text-summarizer.git
cd hybrid-text-summarizer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
