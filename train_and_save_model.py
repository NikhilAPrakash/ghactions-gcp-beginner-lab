import os
import argparse
from datetime import datetime, timezone

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def download_data():
  data = load_breast_cancer()
  features = pd.DataFrame(data.data, columns=data.feature_names)
  target = pd.Series(data.target)
  return features, target


def preprocess_data(X, y):
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  return X_train, X_test, y_train, y_test



def train_model(X_train, y_train):
    """
    Train an ExtraTrees (extremely randomized trees) classifier.
    Similar family to RandomForest, usually faster with strong accuracy.
    """
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model_to_gcs(model, bucket_name, blob_name):
  joblib.dump(model, "model.joblib")
  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  blob.upload_from_filename('model.joblib')

def main():
  X, y = download_data()
  X_train, X_test, y_train, y_test = preprocess_data(X, y)

  model = train_model(X_train, y_train)

  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Model accuracy: {accuracy}')

  bucket_name = "lab-bucket_ghactions"
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  blob_name = f"trained_models/model_{timestamp}.joblib"
  save_model_to_gcs(model, bucket_name, blob_name)
  print(f"Model saved to gs://{bucket_name}/{blob_name}")

if __name__ == "__main__":
  main()

  
