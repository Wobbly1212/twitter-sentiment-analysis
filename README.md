# 🐦 Sentiment Analysis on Twitter Data using Apache Spark

This project performs large-scale **sentiment analysis** on tweets using **Apache Spark** and the **Sentiment140** dataset, implemented and run in **Databricks** ,under the supervision of **Professor Giancarlo Sperli**.

---

## 📚 Project Summary

People express their emotions daily through tweets — short, rich snippets of opinion. This project explores how we can use big data tools to automatically classify those tweets as **positive** or **negative** using machine learning.

We trained and evaluated two machine learning models:
- Logistic Regression  
- Naive Bayes  

Both were implemented with **Spark MLlib** on a dataset of **1.6 million tweets** using Databricks for scalable computation.

---

## 🔍 Dataset

**Sentiment140 Dataset**  
- 1.6 million labeled tweets  
- Balanced: 800,000 positive, 800,000 negative  
- Fields used: `text`, `target (label)`  
- Labels:  
  - `0` = Negative  
  - `4` = Positive → Converted to `1` for binary classification

---

## 🧼 Preprocessing Pipeline

Implemented in PySpark:
- Lowercasing all text
- Removing URLs, mentions, hashtags, punctuation
- Trimming extra whitespace
- Tokenization (word splitting)
- Stopword removal
- **TF-IDF vectorization**

---

## 🧠 Model Training

Split:  
- Training: 80% (1,280,209 tweets)  
- Testing: 20% (319,791 tweets)

### ✅ Logistic Regression
- Accuracy: **0.7796**
- F1 Score: **0.7795**

### ✅ Naive Bayes
- Accuracy: **0.7637**
- F1 Score: **0.7637**

---

## 📊 Model Comparison

| Metric     | Logistic Regression | Naive Bayes |
|------------|---------------------|-------------|
| Accuracy   | 0.7796              | 0.7637      |
| F1 Score   | 0.7795              | 0.7637      |
| Precision  | 0.7797              | 0.7638      |
| Recall     | 0.7796              | 0.7637      |

📌 **Logistic Regression** showed better performance across all metrics, while **Naive Bayes** was faster and simpler to train.

---

## 💻 Tools and Frameworks

- [Apache Spark](https://spark.apache.org/)
- [Databricks](https://databricks.com/)
- PySpark (Spark SQL, MLlib)
- TF-IDF (Term Frequency–Inverse Document Frequency)

---

## 📁 Project Structure

```bash
├── Sentiment140_Spark.html         # Exported Databricks notebook (HTML)
├── Report.pdf                      # Full written report for academic submission
├── README.md                       # This file
