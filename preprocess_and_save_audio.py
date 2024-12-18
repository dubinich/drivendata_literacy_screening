from loguru import logger
import pandas as pd
import torch
import os
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm

DATA_PATH = "/mnt/c/Users/artem/Documents/Projects/drivendata_literacy/data"
AUDIO_PATH = DATA_PATH + "/audio"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(DEVICE)


def load_train_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path + "/train_metadata.csv")


def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze(0), sample_rate


def preprocess_audio(waveform, sample_rate):
    waveform = waveform / torch.max(torch.abs(waveform))  # Normalize
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    return inputs.input_values.to(DEVICE)


def transcribe_audio(input_values):
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])


def main():
    train_meta = load_train_data(DATA_PATH)

    results = []
    for _, row in tqdm(train_meta.iterrows(), total=len(train_meta), desc="Processing Audio Files"):
        filename = row["filename"]
        expected = row["expected_text"]
        file_path = os.path.join(AUDIO_PATH, filename)

        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue

        waveform, sample_rate = load_audio(file_path)
        input_values = preprocess_audio(waveform, sample_rate)
        transcription = transcribe_audio(input_values)

        # Add transcription to results
        row["extracted_phonemes"] = transcription
        results.append(row)

    # Save updated metadata with phonemes
    output_csv = os.path.join(DATA_PATH, "train_metadata_with_phonemes.csv")
    updated_metadata = pd.DataFrame(results)
    updated_metadata.to_csv(output_csv, index=False)

    print(f"Processing completed. Results saved to {output_csv}")


if __name__ == "__main__":
    main()