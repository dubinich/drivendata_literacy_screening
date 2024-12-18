import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
import xgboost as xgb

DATA_PATH = "/mnt/c/Users/artem/Documents/Projects/drivendata_literacy/data"
MODEL_PATH = "/mnt/c/Users/artem/Documents/Projects/drivendata_literacy/models/v0"

os.makedirs(MODEL_PATH, exist_ok=True)

def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance manually."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compute_linguistic_features(row):
    """Compute linguistic features for a single row."""
    extracted_phonemes = row.get("extracted_phonemes", "")
    expected_phonemes = row.get("phonemes", "")

    if not isinstance(extracted_phonemes, str):
        extracted_phonemes = ""
    if not isinstance(expected_phonemes, str):
        expected_phonemes = ""

    if not extracted_phonemes or not expected_phonemes:
        return pd.Series({
            "levenshtein_distance": np.nan,
            "normalized_distance": np.nan,
            "phoneme_overlap_ratio": np.nan,
            "phoneme_coverage": np.nan
        })

    # Clean up phonemes from string to list
    extracted_phonemes = extracted_phonemes.replace("[", "").replace("]", "").replace("'", "").split(", ")
    expected_phonemes = expected_phonemes.replace("[", "").replace("]", "").replace("'", "").split(", ")

    # Levenshtein Distance
    lev_dist = levenshtein_distance(" ".join(extracted_phonemes), " ".join(expected_phonemes))

    # Normalized Distance
    norm_dist = lev_dist / max(len(expected_phonemes), 1)  # Avoid division by zero

    # Phoneme Overlap
    overlap = len(set(extracted_phonemes) & set(expected_phonemes))
    union = len(set(extracted_phonemes) | set(expected_phonemes))
    phoneme_overlap_ratio = overlap / union if union > 0 else 0

    # Phoneme Coverage
    coverage = overlap / len(expected_phonemes) if len(expected_phonemes) > 0 else 0

    return pd.Series({
        "levenshtein_distance": lev_dist,
        "normalized_distance": norm_dist,
        "phoneme_overlap_ratio": phoneme_overlap_ratio,
        "phoneme_coverage": coverage
    })

def read_and_join_data(data_path: str) -> pd.DataFrame:
    main_df = pd.read_csv(data_path + "/train_metadata.csv")
    df_with_audio = pd.read_csv(data_path + "/train_metadata_with_phonemes.csv")
    df_with_text_to_phoneme = pd.read_csv(data_path + "/train_metadata_with_text_to_phonemes.csv")
    labels = pd.read_csv(data_path + "/train_labels.csv")

    main_df = main_df.merge(
        df_with_audio, how="left", on=["filename", "task","expected_text","grade"])
    main_df = main_df.merge(
        df_with_text_to_phoneme, how="left", on=["filename", "task","expected_text","grade"])
    
    main_df = main_df.merge(labels, how="left", on="filename")

    return main_df


def train_xgboost(train_df, model_path):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cols_to_ignore = ["filename", "score", "fold"]
    categorical_cols = ["task", "grade", "expected_text"]
    phoneme_cols = ["extracted_phonemes", "phonemes"]

    cols_to_ignore += categorical_cols + phoneme_cols
    train_cols = [col for col in train_df.columns if col not in cols_to_ignore]
    X = train_df[train_cols]
    y = train_df["score"]
    groups = train_df["expected_text"]

    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
        print(f"Training Fold {fold}...")

        # Split train and test
        train_fold = train_df.iloc[train_index]
        test_fold = train_df.iloc[test_index]

        X_train = train_fold[train_cols]
        y_train = train_fold["score"]
        X_test = test_fold[train_cols]
        y_test = test_fold["score"]

        # Initialize XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # Adjust based on your task
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

        # Predict on test fold
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

        # Compute log loss
        fold_logloss = log_loss(y_test, y_pred_prob)
        print(f"Fold {fold} Log Loss: {fold_logloss}")
        fold_metrics.append(fold_logloss)

        # Save the model
        fold_model_path = os.path.join(model_path, f"xgboost_fold_{fold}.joblib")
        joblib.dump(model, fold_model_path)
        print(f"Model for Fold {fold} saved at: {fold_model_path}")

    # Print overall metrics
    print(f"Mean Log Loss Across Folds: {np.mean(fold_metrics)}")


def main():
    # Assuming `read_and_join_data` and `compute_linguistic_features` are already defined
    train_df = read_and_join_data(DATA_PATH)
    
    linguistic_features = train_df.apply(compute_linguistic_features, axis=1)
    train_df = pd.concat([train_df, linguistic_features], axis=1)

    # Train XGBoost and save models
    train_xgboost(train_df, MODEL_PATH)

if __name__ == "__main__":
    main()