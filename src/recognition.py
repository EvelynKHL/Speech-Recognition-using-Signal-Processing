import os
import joblib
from src.feature_extraction import extract_features

MODEL_FILENAME = "model/speech_recognition_model.pkl"


def recognize_speech(file_path):

    if not os.path.exists(MODEL_FILENAME):
        return "Model not found"

    trained_models = joblib.load(MODEL_FILENAME)

    features = extract_features(file_path)

    best_score = float('-inf')
    best_label = None

    for label, model in trained_models.items():

        try:
            score = model.score(features)

            if score > best_score:
                best_score = score
                best_label = label

        except Exception as e:
            print(e)

    return best_label
