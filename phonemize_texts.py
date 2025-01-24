from loguru import logger
import numpy as np
import pandas as pd
import torch
import os
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

DATA_PATH = "data"
AUDIO_PATH = DATA_PATH + "/audio"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHONEMIZER_PATH = os.path.join(DATA_PATH, "phonemizer")
os.makedirs(PHONEMIZER_PATH, exist_ok=True)


def load_train_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path + "/train_metadata.csv")
    return df


def main():
    train_meta = load_train_data(DATA_PATH)

    txt_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100').to(DEVICE)
    txt_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    txt_model.save_pretrained(PHONEMIZER_PATH)
    txt_tokenizer.save_pretrained(PHONEMIZER_PATH)

    results = []
    for _, row in tqdm(train_meta.iterrows(), total=len(train_meta), desc="Processing texts"):
        words_txt = [f'<eng-us>: {word}' for word in row["expected_text"].split()]
        out = txt_tokenizer(
            words_txt, padding=True, add_special_tokens=False, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            preds = txt_model.generate(**out, num_beams=1, max_length=50)
        
        phonemes = txt_tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)

        row["phonemes"] = phonemes
        results.append(row)

    output_csv = os.path.join(DATA_PATH, "train_metadata_with_text_to_phonemes.csv")
    updated_metadata = pd.DataFrame(results)
    updated_metadata.to_csv(output_csv, index=False)

    print(f"Processing completed. Results saved to {output_csv}")


if __name__ == "__main__":
    main()

