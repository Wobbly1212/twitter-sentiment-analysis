# Sentiment Analysis on Twitter Data Using Apache Spark

Large-scale **sentiment analysis** on 1.6 million tweets using **Apache Spark** and the **Sentiment140** dataset, implemented in **Databricks** under the supervision of **Professor Giancarlo Sperli**.

## Project Summary

People express their emotions daily through tweets — short, rich snippets of opinion. This project explores how big data tools can automatically classify tweets as **positive** or **negative** using machine learning.

Two models were trained and evaluated:
- **Logistic Regression**
- **Naive Bayes**

Both were implemented with **Spark MLlib** on a dataset of **1.6 million tweets** using Databricks for scalable computation.

## Results

| Metric | Logistic Regression | Naive Bayes |
|--------|---------------------|-------------|
| Accuracy | 0.7796 | 0.7637 |
| F1 Score | 0.7795 | 0.7637 |
| Precision | 0.7797 | 0.7638 |
| Recall | 0.7796 | 0.7637 |

**Logistic Regression** outperformed across all metrics, while **Naive Bayes** was faster to train.

## Dataset

**[Sentiment140](http://help.sentiment140.com/for-students)**
- 1.6 million labeled tweets (balanced: 800K positive, 800K negative)
- Labels: `0` = Negative, `4` = Positive (converted to `1`)
- Fields used: `text`, `target`

## Preprocessing Pipeline

Implemented in PySpark:

1. Lowercasing all text
2. Removing URLs, mentions, hashtags, punctuation
3. Trimming extra whitespace
4. Tokenization
5. Stopword removal
6. **TF-IDF vectorization**

## Getting Started

### Prerequisites

- Python 3.10+
- Apache Spark 3.4+ or a [Databricks](https://databricks.com/) account

### Installation

```bash
git clone https://github.com/Wobbly1212/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -r requirements.txt
```

### Running

The analysis was originally run on Databricks. The exported notebook is available in `docs/`:
- `Sentiment140_Spark.html` — Full Databricks notebook export
- `Report.pdf` — Academic report with methodology and analysis

To run locally, you'll need a Spark environment configured.

## Project Structure

```
twitter-sentiment-analysis/
├── docs/
│   ├── Sentiment140_Spark.html   # Exported Databricks notebook
│   └── Report.pdf                # Academic report
├── requirements.txt
├── LICENSE
└── README.md
```

## Tools & Frameworks

- [Apache Spark](https://spark.apache.org/) / PySpark
- [Databricks](https://databricks.com/)
- Spark MLlib (Logistic Regression, Naive Bayes)
- TF-IDF (Term Frequency-Inverse Document Frequency)

## Author

Developed by **Diako Darabi** as part of the Data Science Master's Program.

## License

This project is licensed under the [MIT License](LICENSE).
