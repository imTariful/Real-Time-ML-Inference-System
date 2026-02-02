import os
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import onnxmltools

# Create directories
os.makedirs("models/sentiment_v1", exist_ok=True)
os.makedirs("models/sentiment_v2", exist_ok=True)

# Fake data
corpus = [
    "I love this product",
    "This is terrible",
    "Great service",
    "Worst experience ever",
    "Highly recommended",
    "Do not buy this",
]
labels = [1, 0, 1, 0, 1, 0]

# --- Model V1: Logistic Regression ---
print("Training Model V1 (Logistic Regression)...")
pipeline_v1 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
pipeline_v1.fit(corpus, labels)

# Convert to ONNX
initial_type = [('input', StringTensorType([None, 1]))]
onnx_model_v1 = convert_sklearn(pipeline_v1, initial_types=initial_type, target_opset=12)

with open("models/sentiment_v1/model.onnx", "wb") as f:
    f.write(onnx_model_v1.SerializeToString())
print("Saved models/sentiment_v1/model.onnx")

# --- Model V2: Random Forest ---
print("Training Model V2 (Random Forest)...")
pipeline_v2 = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=10))
])
pipeline_v2.fit(corpus, labels)

# Convert to ONNX
onnx_model_v2 = convert_sklearn(pipeline_v2, initial_types=initial_type, target_opset=12)

with open("models/sentiment_v2/model.onnx", "wb") as f:
    f.write(onnx_model_v2.SerializeToString())
print("Saved models/sentiment_v2/model.onnx")

print("Dummy models generated successfully.")
