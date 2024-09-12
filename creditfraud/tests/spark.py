import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create a Spark session
spark = SparkSession.builder.appName("SparkVsSklearn").getOrCreate()

# Generate a large dataset
print("Generating dataset...")
n_samples = 1_000_000
n_features = 10
np.random.seed(42)
X = np.random.rand(n_samples, n_features)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PySpark implementation
print("Running PySpark implementation...")
spark_start = time.time()

try:
    # Convert to Spark DataFrame
    train_data = spark.createDataFrame(
        pd.DataFrame(np.column_stack((X_train, y_train)), 
                     columns=[f'feature_{i}' for i in range(n_features)] + ['label'])
    )
    test_data = spark.createDataFrame(
        pd.DataFrame(np.column_stack((X_test, y_test)), 
                     columns=[f'feature_{i}' for i in range(n_features)] + ['label'])
    )

    # Prepare features
    assembler = VectorAssembler(inputCols=[f'feature_{i}' for i in range(n_features)], outputCol="features")
    train_data = assembler.transform(train_data)
    test_data = assembler.transform(test_data)

    # Train and evaluate
    rf = SparkRF(numTrees=10, maxDepth=5, labelCol="label", featuresCol="features", seed=42)
    model = rf.fit(train_data)
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    spark_accuracy = evaluator.evaluate(predictions)

    spark_end = time.time()
    spark_time = spark_end - spark_start

    print(f"PySpark Accuracy: {spark_accuracy:.4f}")
    print(f"PySpark Time: {spark_time:.2f} seconds")

except Exception as e:
    print(f"Error in PySpark implementation: {str(e)}")
    spark_accuracy = None
    spark_time = None

# Scikit-learn implementation
print("\nRunning Scikit-learn implementation...")
sklearn_start = time.time()

try:
    clf = SklearnRF(n_estimators=10, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, y_pred)

    sklearn_end = time.time()
    sklearn_time = sklearn_end - sklearn_start

    print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")
    print(f"Scikit-learn Time: {sklearn_time:.2f} seconds")

except Exception as e:
    print(f"Error in Scikit-learn implementation: {str(e)}")
    sklearn_accuracy = None
    sklearn_time = None

# Print comparison results
print("\nComparison Results:")
if spark_accuracy is not None and sklearn_accuracy is not None:
    print(f"PySpark Accuracy: {spark_accuracy:.4f}")
    print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")
    print(f"PySpark Time: {spark_time:.2f} seconds")
    print(f"Scikit-learn Time: {sklearn_time:.2f} seconds")
    if spark_time > 0:
        print(f"Speedup factor: {sklearn_time / spark_time:.2f}x")
    else:
        print("Speedup factor: N/A (PySpark time is 0 or negative)")
else:
    print("Unable to compare due to errors in execution.")

# Clean up
spark.stop()