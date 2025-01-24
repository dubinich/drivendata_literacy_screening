import os
import string
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import librosa
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher
from tqdm import tqdm
import opensmile

tqdm.pandas()

DATA_PATH = "data"
MODEL_PATH = "models/v1"
TRAINING_DATA_SUBFOLDER = os.path.join(DATA_PATH, "training_data")
TFIDF_PATH = os.path.join(DATA_PATH, "vectorizer_v1")
SVD_PATH = os.path.join(DATA_PATH, "svd_transformer_v1")
os.makedirs(TRAINING_DATA_SUBFOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TFIDF_PATH, exist_ok=True)
os.makedirs(SVD_PATH, exist_ok=True)


def clean_text(input_text):
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = input_text.translate(translator).lower()
    return cleaned_text


def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_cer(reference, hypothesis):
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")
    return calculate_wer(ref_chars, hyp_chars)


def ngram_precision_recall(expected_text, transcription, n=2):
    expected_words = expected_text.split()
    transcription_words = transcription.split()
    
    expected_ngrams = [tuple(expected_words[i:i + n]) for i in range(len(expected_words) - n + 1)]
    transcription_ngrams = [tuple(transcription_words[i:i + n]) for i in range(len(transcription_words) - n + 1)]
    
    expected_set = set(expected_ngrams)
    transcription_set = set(transcription_ngrams)

    precision = len(expected_set & transcription_set) / len(transcription_set) if transcription_set else 0
    recall = len(expected_set & transcription_set) / len(expected_set) if expected_set else 0
    return precision, recall


def compute_features_for_whisper(row):
    transcription = row.get("transcription", "")
    expected_text = row.get("expected_text", "")

    filtered_transcription = clean_text(str(transcription))
    filtered_expected_text = clean_text(str(expected_text))

    similarity = SequenceMatcher(None, filtered_transcription, filtered_expected_text, autojunk=False).ratio()

    # WER
    wer = calculate_wer(filtered_expected_text, filtered_transcription)

    # CER
    cer = calculate_cer(filtered_expected_text, filtered_transcription)

    # Ngram P&R
    bigram_precision, bigram_recall = ngram_precision_recall(filtered_expected_text, filtered_transcription, 2)
    trigram_precision, trigram_recall = ngram_precision_recall(filtered_expected_text, filtered_transcription, 3)
    features = {
        "seq_similarity": similarity,
        "wer": wer,
        "cer": cer,
        "bigram_precision": bigram_precision,
        "bigram_recall": bigram_recall,
        "trigram_precision": trigram_precision,
        "trigram_recall": trigram_recall,
    }
    return pd.Series(features)


def load_phoneme_dict(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        return json.load(f)


def phonetic_complexity(word, phoneme_dict):
    word_lower = word.lower()
    if word_lower not in phoneme_dict:
        return len(word)
    
    phonemes = phoneme_dict[word_lower][0].split()
    phoneme_count = len(phonemes)
    cluster_count = sum(1 for phoneme in phonemes if len(phoneme) > 2 and phoneme[-1].isdigit())
    return phoneme_count + cluster_count
        

def word_familiarity(word):
    """Estimate word familiarity by checking simple heuristic."""
    return 1 if len(word) <= 5 else 0


def sentence_length(sentence):
    return len(sentence.split())


def lexical_complexity(word):
    return max(1, len(word) // 3)


def compute_features_for_literacy_task(row, phoneme_dict):
    sentence = row.get("expected_text", "")
    filtered_sentence = clean_text(str(sentence))
    words = filtered_sentence.split()
    phonetic_score = sum(phonetic_complexity(word, phoneme_dict) for word in words) / (len(words)+1)
    familiarity_score = sum(word_familiarity(word) for word in words) / (len(words)+1)
    length_score = sentence_length(filtered_sentence)
    lexical_score = sum(lexical_complexity(word) for word in words) / (len(words)+1)

    features = {
        "phonetic_complexity": phonetic_score,
        "word_familiarity": familiarity_score,
        "sentence_length": length_score,
        "lexical_complexity": lexical_score,
    }
    return pd.Series(features)


def compute_phoneme_alignment(row):
    extracted_phonemes = row.get("extracted_phonemes", "")
    expected_phonemes = row.get("phonemes", "")
    
    if not extracted_phonemes or not expected_phonemes:
        return pd.Series({"phoneme_alignment_score": np.nan})
    
    try:
        alignment_distance, _ = librosa.sequence.dtw(
            np.array([ord(ch) for ch in expected_phonemes]),
            np.array([ord(ch) for ch in extracted_phonemes])
        )
        alignment_score = -np.mean(alignment_distance)
    except Exception as e:
        alignment_score = np.nan

    return pd.Series({"phoneme_alignment_score": alignment_score})


def extract_audio_features(audio_path, smile):
    features = smile.process_file(audio_path)
    return features.mean(axis=0)


def enrich_features(train_df):
    print("Starting to enrich features...")

    print("Computing linguistic features...")
    linguistic_features = train_df.apply(compute_linguistic_features, axis=1)
    train_df = pd.concat([train_df, linguistic_features], axis=1)
    print("Linguistic features added.")

    print("Computing basic text features...")
    text_features = train_df.apply(compute_basic_text_features, axis=1)
    train_df = pd.concat([train_df, text_features], axis=1)
    print("Basic text features added.")

    print("Extracting audio features...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feature_list = []
    for filename in tqdm(train_df.filename, desc="Extracting OpenSMILE Features", unit="file"):
        features = smile.process_file(os.path.join(DATA_PATH, "audio", filename))  # Extract features for each file
        feature_list.append(features.mean(axis=0))
    audio_features = pd.DataFrame(feature_list)
    train_df = pd.concat([train_df, audio_features], axis=1)
    print("Audio features added.")

    print("Computing phoneme alignment features...")
    alignment_features = train_df.apply(compute_phoneme_alignment, axis=1)
    train_df = pd.concat([train_df, alignment_features], axis=1)
    print("Phoneme alignment features added.")

    print("Extracting whisper features...")
    whisper_features = train_df.apply(compute_features_for_whisper, axis=1)
    train_df = pd.concat([train_df, whisper_features], axis=1)
    print("Whisper features added.")

    phoneme_dict = load_phoneme_dict('data/cmudict.json') # TODO: path as an argument.
    complexity_features = train_df.apply(lambda x: compute_features_for_literacy_task(x, phoneme_dict), axis=1)
    train_df = pd.concat([train_df, complexity_features], axis=1)
    print("Feature enrichment completed.")
    return train_df


def read_and_join_data(data_path: str) -> pd.DataFrame:
    main_df = pd.read_csv(data_path + "/train_metadata.csv")
    df_with_audio = pd.read_csv(data_path + "/train_metadata_with_phonemes.csv")
    df_with_text_to_phoneme = pd.read_csv(data_path + "/train_metadata_with_text_to_phonemes.csv")
    df_with_whisper = pd.read_csv(data_path + "/train_metadata_with_whisper.csv")
    df_with_whisper = df_with_whisper[["filename", "task","expected_text","grade", "transcription"]]
    labels = pd.read_csv(data_path + "/train_labels.csv")

    main_df = main_df.merge(
        df_with_audio, how="left", on=["filename", "task","expected_text","grade"])
    main_df = main_df.merge(
        df_with_text_to_phoneme, how="left", on=["filename", "task","expected_text","grade"])
    main_df = main_df.merge(
        df_with_whisper, how="left", on=["filename", "task","expected_text","grade"])
    
    main_df = main_df.merge(labels, how="left", on="filename")

    return main_df


def compute_linguistic_features(row):
    extracted_phonemes = row.get("extracted_phonemes", "")
    expected_phonemes = row.get("phonemes", "")

    if not isinstance(extracted_phonemes, str) or not isinstance(expected_phonemes, str):
        return pd.Series({
            "levenshtein_distance": np.nan,
            "normalized_distance": np.nan,
            "phoneme_overlap_ratio": np.nan,
            "phoneme_coverage": np.nan,
            "num_extracted_phonemes": np.nan,
            "num_expected_phonemes": np.nan,
            "diff_num_phonemes": np.nan,
            "ratio_num_phonemes": np.nan,
            "num_insertions": np.nan,
            "num_deletions": np.nan,
            "num_substitutions": np.nan,
            "lcs_ratio": np.nan
        })

    extracted_phonemes = extracted_phonemes.replace("[", "").replace("]", "").replace("'", "").split(" ")
    expected_phonemes = expected_phonemes.replace("[", "").replace("]", "").replace("'", "").split(", ")

    extracted_phonemes = "".join([x.strip() for x in extracted_phonemes])
    expected_phonemes = "".join([x.strip() for x in expected_phonemes])

    # Basic counts
    num_extracted = len(extracted_phonemes)
    num_expected = len(expected_phonemes)

    # Levenshtein distance
    lev_dist = levenshtein_distance(expected_phonemes, extracted_phonemes)

    # Normalized distance
    norm_dist = lev_dist / max(num_expected, 1)

    # Phoneme overlap
    overlap = len(set(extracted_phonemes) & set(expected_phonemes))
    union = len(set(extracted_phonemes) | set(expected_phonemes))
    phoneme_overlap_ratio = overlap / union if union > 0 else 0

    # Phoneme coverage
    coverage = overlap / num_expected if num_expected > 0 else 0
    num_deletions = max(0, num_expected - num_extracted)
    lcs_length = num_expected - num_deletions
    lcs_ratio = lcs_length / num_expected if num_expected > 0 else 0

    return pd.Series({
        "levenshtein_distance": lev_dist,
        "normalized_distance": norm_dist,
        "phoneme_overlap_ratio": phoneme_overlap_ratio,
        "phoneme_coverage": coverage,
        "num_extracted_phonemes": num_extracted,
        "num_expected_phonemes": num_expected,
        "diff_num_phonemes": abs(num_extracted - num_expected),
        "ratio_num_phonemes": num_extracted / num_expected if num_expected > 0 else np.nan,
        "num_insertions": max(0, num_extracted - num_expected),
        "num_deletions": max(0, num_expected - num_extracted),
        "num_substitutions": lev_dist - abs(num_extracted - num_expected),
        "lcs_ratio": lcs_ratio
    })


def compute_basic_text_features(row):
    expected_text = row.get("expected_text", "")

    if not isinstance(expected_text, str):
        expected_text = ""

    text_len = len(expected_text)
    num_words = len(expected_text.split())
    num_vowels = sum(1 for char in expected_text.lower() if char in "aeiouy")

    vowel_ratio = num_vowels / text_len if text_len > 0 else 0
    avg_word_len = text_len / num_words if num_words > 0 else 0

    return pd.Series({
        "text_len": text_len,
        "num_words": num_words,
        "num_vowels": num_vowels,
        "vowel_ratio": vowel_ratio,
        "avg_word_len": avg_word_len
    })


def target_encode(train_df, categorical_cols, target_col, fold_col=None, smoothing_factor=10, min_samples_leaf=10):
    encoded_df = train_df.copy()

    for col in categorical_cols:
        if fold_col:
            encoded_df[f'{col}_target_encoded'] = np.nan
            for fold in train_df[fold_col].unique():
                train_idx = train_df[fold_col] != fold
                val_idx = train_df[fold_col] == fold

                global_mean = train_df.loc[train_idx, target_col].mean()
                category_means = train_df.loc[train_idx].groupby(col)[target_col].agg(['mean', 'count'])

                smoothing = 1 / (1 + np.exp(-(category_means['count'] - min_samples_leaf) / smoothing_factor))
                encoded_values = smoothing * category_means['mean'] + (1 - smoothing) * global_mean

                encoded_df.loc[val_idx, f'{col}_target_encoded'] = train_df.loc[val_idx, col].map(encoded_values)
        else:
            global_mean = train_df[target_col].mean()
            category_means = train_df.groupby(col)[target_col].agg(['mean', 'count'])

            smoothing = 1 / (1 + np.exp(-(category_means['count'] - min_samples_leaf) / smoothing_factor))
            encoded_values = smoothing * category_means['mean'] + (1 - smoothing) * global_mean

            encoded_df[f'{col}_target_encoded'] = train_df[col].map(encoded_values)

    return encoded_df


def add_tfidf_svd_features(df, max_features=25000, n_components=4):
    """Add SVD-compressed TF-IDF features (for chars ngrams)."""
    vectorizer = TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(df['expected_text'].fillna(""))

    tfidf_vectorizer_path = os.path.join(TFIDF_PATH, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, tfidf_vectorizer_path)
    print(f"TFIDF Vectorizer saved at: {tfidf_vectorizer_path}")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_features = svd.fit_transform(tfidf_matrix)

    svd_transformer_path = os.path.join(SVD_PATH, "svd_transformer.pkl")
    joblib.dump(svd, svd_transformer_path)
    print(f"SVD Transformer saved at: {svd_transformer_path}")

    svd_feature_names = [f'svd_feature_{i+1}' for i in range(n_components)]
    svd_df = pd.DataFrame(svd_features, columns=svd_feature_names)
    return pd.concat([df, svd_df], axis=1) 


def train_xgboost_with_calibration(train_df, model_path, calibrate=False):
    # Jus a regular stratified kfold.
    sgkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Using expected test as a group to ensure model generalizes for unseen data.
    # sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    cols_to_ignore = ["filename", "score"]
    categorical_cols = ["task", "grade", "expected_text", "transcription"]
    phoneme_cols = ["extracted_phonemes", "phonemes"]

    X = train_df.drop(columns=cols_to_ignore + categorical_cols + phoneme_cols)
    y = train_df["score"]
    # group = train_df["expected_text"]

    train_df["fold"] = -1
    for fold, (_, test_index) in enumerate(sgkf.split(X, y)): #, group
        train_df.loc[test_index, "fold"] = fold

    # Perform target encoding
    train_df = target_encode(train_df, categorical_cols, "score", fold_col="fold")
    for col in categorical_cols:
        train_df[f"{col}_freq_encoded"] = train_df[col].map(train_df[col].value_counts(normalize=True))

    # Add SVD features
    train_df = add_tfidf_svd_features(train_df)

    # Update columns for training
    encoded_cols = [f'{col}_target_encoded' for col in categorical_cols]
    cols_to_ignore += categorical_cols + phoneme_cols + ["fold"]
    train_cols = [col for col in train_df.columns if col not in cols_to_ignore]

    kfold_predictions = np.zeros(len(train_df))
    fold_metrics = []

    for fold in sorted(train_df['fold'].unique()):
        print(f"Training Fold {fold}...")

        train_index = train_df[train_df['fold'] != fold].index
        test_index = train_df[train_df['fold'] == fold].index
        train_fold = train_df.iloc[train_index]
        test_fold = train_df.iloc[test_index]

        X_train = train_fold[train_cols]
        y_train = train_fold["score"]
        X_test = test_fold[train_cols]
        y_test = test_fold["score"]

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            learning_rate=0.01,
            n_estimators=1500,
            max_depth=15,
            colsample_bytree=0.5,
            subsample=0.5,
            early_stopping_rounds=75,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        if calibrate:
            print(f"Calibrating Fold {fold}...")
            calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated_model.fit(X_test, y_test)
            model = calibrated_model

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        kfold_predictions[test_index] = y_pred_prob

        fold_logloss = log_loss(y_test, y_pred_prob)
        print(f"Fold {fold} Log Loss: {fold_logloss}")
        fold_metrics.append(fold_logloss)

        fold_model_path = os.path.join(model_path, f"xgboost_fold_{fold}.joblib")
        joblib.dump(model, fold_model_path)
        print(f"Model for Fold {fold} saved at: {fold_model_path}")

    train_df["kfold_predictions"] = kfold_predictions

    output_path = os.path.join(TRAINING_DATA_SUBFOLDER, "train_with_kfold_predictions.csv")
    train_df.to_csv(output_path, index=False)
    print(f"Training Data with K-Fold Predictions saved at: {output_path}")

    # Print overall metrics
    print(f"Mean Log Loss Across Folds: {np.mean(fold_metrics)}")
    return train_df

# Example usage in main
def main():
    train_df = read_and_join_data(DATA_PATH)
    print(f"Training data shape: {train_df.shape}")

    # print(train_df["transcription"].value_counts())
    train_df = enrich_features(train_df)
    print(train_df.head())

    train_xgboost_with_calibration(train_df, MODEL_PATH, calibrate=False)
    # train_xgboost_with_calibration(train_df, MODEL_PATH, calibrate=True)

if __name__ == "__main__":
    main()