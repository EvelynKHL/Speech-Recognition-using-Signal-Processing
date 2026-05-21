import os
import numpy as np
import joblib
from hmmlearn import hmm
from src.feature_extraction import extract_features

TRAINING_DATA_DIR = "training_data"
MODEL_FILENAME = "model/speech_recognition_model.pkl"


def train_model():
    models = {}

    for file in os.listdir(TRAINING_DATA_DIR):
        if file.endswith(".wav"):
            label = file.split('_')[0]

            file_path = os.path.join(TRAINING_DATA_DIR, file)

            features = extract_features(file_path)

            if label not in models:
                models[label] = []

            models[label].append(features)

    trained_models = {}

    for label, features_list in models.items():
        all_features = np.vstack(features_list)

        model = hmm.GaussianHMM(
            n_components=5,
            covariance_type='diag',
            n_iter=100
        )

        model.fit(all_features)

        trained_models[label] = model

    joblib.dump(trained_models, MODEL_FILENAME)

    print("Model Trained Successfully")
