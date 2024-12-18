from loguru import logger
import numpy as np
import pandas as pd
import torch
import os
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

DATA_PATH = "/mnt/c/Users/artem/Documents/Projects/drivendata_literacy/data"
AUDIO_PATH = DATA_PATH + "/audio"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_train_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path + "/train_metadata.csv")
    return df


def main():
    train_meta = load_train_data(DATA_PATH)

    # Load model and tokenizer, then move the model to GPU
    txt_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100').to(DEVICE)
    txt_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    results = []
    for _, row in tqdm(train_meta.iterrows(), total=len(train_meta), desc="Processing texts"):
        # Tokenize text and move to GPU
        words_txt = [f'<eng-us>: {word}' for word in row["expected_text"].split()]
        out = txt_tokenizer(
            words_txt, padding=True, add_special_tokens=False, return_tensors='pt').to(DEVICE)
        
        # Generate predictions and move model to GPU if necessary
        with torch.no_grad():  # Disable gradients for inference
            preds = txt_model.generate(**out, num_beams=1, max_length=50)
        
        # Decode predictions and move results to CPU before appending to the list
        phonemes = txt_tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)

        row["phonemes"] = phonemes
        results.append(row)

    output_csv = os.path.join(DATA_PATH, "train_metadata_with_text_to_phonemes.csv")
    updated_metadata = pd.DataFrame(results)
    updated_metadata.to_csv(output_csv, index=False)

    print(f"Processing completed. Results saved to {output_csv}")


if __name__ == "__main__":
    main()

