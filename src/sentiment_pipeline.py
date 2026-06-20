"""
Sentiment analysis on the Sentiment140 dataset (1.6M tweets) with PySpark + Spark MLlib.

This is the standalone, runnable form of the Databricks notebook in
``notebooks/sentiment140_spark.ipynb``. The notebook assumes a Databricks
workspace where ``spark`` is pre-defined and the data lives on DBFS; this script
creates its own ``SparkSession`` so it can run anywhere via ``spark-submit`` or
plain ``python``.

Pipeline:
    load CSV -> clean text -> tokenize -> remove stopwords -> TF-IDF
    -> train Logistic Regression and Naive Bayes -> evaluate (accuracy/F1/precision/recall)

Usage:
    spark-submit src/sentiment_pipeline.py --input path/to/training.1600000.processed.noemoticon.csv
    # or, for a quick local smoke test on a small sample:
    python src/sentiment_pipeline.py --input sample.csv --sample 0.001

Dataset: http://help.sentiment140.com/for-students  (6 columns, no header:
    target, id, date, query, user, text ; target 0 = negative, 4 = positive)
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, trim, when, count
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SENTIMENT140_COLUMNS = ["target", "id", "date", "query", "user", "text"]


def load_data(spark, file_path):
    """Load the raw Sentiment140 CSV (6 columns, no header) and name the columns."""
    df = (
        spark.read.format("csv")
        .option("header", "false")
        .option("inferSchema", "false")
        .option("sep", ",")
        .load(file_path)
    )
    return df.toDF(*SENTIMENT140_COLUMNS).cache()


def explore(df):
    """Print total count, null counts, and class balance."""
    print(f"Total tweets: {df.count():,}")
    df.select([count(when(col(c).isNull(), c)).alias(f"{c}_nulls") for c in ("target", "text")]).show()
    print("Class balance (target):")
    df.groupBy("target").count().orderBy("target").show()


def clean_text(df):
    """Remove URLs, @mentions, hashtags and punctuation, lowercase, squeeze whitespace."""
    return (
        df.select("target", "text")
        .withColumn(
            "text_clean",
            lower(regexp_replace(col("text"), r"http\S+|@\S+|#[A-Za-z0-9_]+|[^a-zA-Z\s]", " ")),
        )
        .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " "))
        .na.drop()
    )


def build_features(df_clean, num_features=100_000):
    """Tokenize, drop stopwords, then TF-IDF vectorize into a `features` column."""
    df_tokens = df_clean.withColumn("words", split(trim(col("text_clean")), " +"))
    df_tokens = StopWordsRemover(inputCol="words", outputCol="filtered_words").transform(df_tokens)

    tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=num_features)
    df_tf = tf.transform(df_tokens)

    idf_model = IDF(inputCol="rawFeatures", outputCol="features").fit(df_tf)
    df_tfidf = idf_model.transform(df_tf)

    # 0 = negative, 4 = positive  ->  0 / 1
    return df_tfidf.withColumn("label", when(col("target") == 4, 1).otherwise(0))


def evaluate(predictions, name):
    """Print accuracy, F1, weighted precision and weighted recall for a prediction set."""
    metrics = {
        "Accuracy": "accuracy",
        "F1 Score": "f1",
        "Precision": "weightedPrecision",
        "Recall": "weightedRecall",
    }
    print(f"\n{name} performance:")
    results = {}
    for label, metric_name in metrics.items():
        score = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric_name
        ).evaluate(predictions)
        results[label] = score
        print(f"  {label:<9}: {score:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Sentiment140 sentiment analysis with PySpark")
    parser.add_argument("--input", required=True, help="Path to the Sentiment140 CSV")
    parser.add_argument("--num-features", type=int, default=100_000, help="HashingTF feature dimension")
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction to sample (for quick local runs)")
    parser.add_argument("--save-model", default=None, help="Optional path to save the trained LR model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Sentiment140-Spark").getOrCreate()
    try:
        df = load_data(spark, args.input)
        if args.sample < 1.0:
            df = df.sample(withReplacement=False, fraction=args.sample, seed=args.seed)

        explore(df)
        df_features = build_features(clean_text(df), num_features=args.num_features)

        train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=args.seed)
        print(f"\nTraining rows: {train_data.count():,} | Testing rows: {test_data.count():,}")

        lr_model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20).fit(train_data)
        evaluate(lr_model.transform(test_data), "Logistic Regression")

        nb_model = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial").fit(train_data)
        evaluate(nb_model.transform(test_data), "Naive Bayes")

        if args.save_model:
            lr_model.write().overwrite().save(args.save_model)
            print(f"\nLogistic Regression model saved to {args.save_model}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
