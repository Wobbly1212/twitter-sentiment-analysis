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

The pipeline ships as runnable code in two forms:

| File | Use it when |
|------|-------------|
| [`notebooks/sentiment140_spark.ipynb`](notebooks/sentiment140_spark.ipynb) | You want the full annotated analysis (EDA, plots, both models). Assumes a Databricks workspace where `spark` is pre-defined and the CSV is on DBFS. |
| [`src/sentiment_pipeline.py`](src/sentiment_pipeline.py) | You want to run it anywhere. Creates its own `SparkSession`, parameterized via CLI flags. |

**On Databricks:** import `notebooks/sentiment140_spark.ipynb`, upload the Sentiment140 CSV to DBFS, and run all cells.

**Standalone with `spark-submit`:**

```bash
pip install -r requirements.txt
# full dataset
spark-submit src/sentiment_pipeline.py --input training.1600000.processed.noemoticon.csv
# quick local smoke test on a 0.1% sample
spark-submit src/sentiment_pipeline.py --input training.1600000.processed.noemoticon.csv --sample 0.001
```

The script loads the data, cleans the text, builds TF-IDF features, trains both Logistic Regression and Naive Bayes, and prints accuracy / F1 / precision / recall for each.

> The notebook and report (`docs/`) are the original Databricks deliverables; the `notebooks/` and `src/` code is reconstructed faithfully from the exported notebook so the project is runnable outside Databricks.

## Project Structure

```
twitter-sentiment-analysis/
├── notebooks/
│   └── sentiment140_spark.ipynb  # annotated analysis (Databricks)
├── src/
│   └── sentiment_pipeline.py     # standalone, spark-submit-able pipeline
├── docs/
│   ├── Sentiment140_Spark.html   # original exported Databricks notebook
│   └── Report.pdf                # academic report
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
