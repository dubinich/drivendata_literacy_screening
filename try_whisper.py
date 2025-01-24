from loguru import logger
import numpy as np
import pandas as pd
import torch
import whisper
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from transformers import T5ForConditionalGeneration, AutoTokenizer
from difflib import SequenceMatcher
from tqdm import tqdm

DATA_PATH = "data"
AUDIO_PATH = DATA_PATH + "/audio"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_train_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path + "/train_metadata.csv")
    return df

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000

    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.squeeze(0), sample_rate

def preprocess_audio(waveform):
    waveform = waveform / torch.max(torch.abs(waveform))
    return waveform

def constrained_decoding(model, audio_file, expected_text, language="english"):
    waveform, sample_rate = load_audio(audio_file)
    waveform = preprocess_audio(waveform)

    vocab = set(expected_text.lower())

    result = model.transcribe(
        audio_file,
        language=language,
        temperature=0.0,
        initial_prompt=f"Transcribe the audio, expected transcription: {expected_text}",
        logprob_threshold=-1.0,
        no_speech_threshold=0.1
    )

    transcription = result["text"]
    filtered_transcription = ''.join([char for char in transcription.lower() if char in vocab]).strip()
    similarity = SequenceMatcher(None, filtered_transcription, expected_text, autojunk=False).ratio()
    constrained_output = filtered_transcription

    return transcription, constrained_output, similarity

def run_constrained_whisper(data, expected_texts):
    model = whisper.load_model("assets/large-v3-turbo.pt").to(DEVICE)
    results = []
    for filename, expected in tqdm(zip(data, expected_texts), total=len(data), desc="Processing Audio Files"):
        transcription, constrained, similarity_score = constrained_decoding(
            model, AUDIO_PATH + f"/{filename}", expected
        )
        results.append((filename, expected, transcription, constrained, similarity_score))
    return results

def main():
    train_meta = load_train_data(DATA_PATH)

    results = run_constrained_whisper(
        train_meta["filename"].tolist(), train_meta["expected_text"].tolist()
    )

    results_df = pd.DataFrame(results, columns=["filename", "expected_text", "transcription", "constrained_output", "similarity_score"])
    train_meta_with_results = pd.merge(train_meta, results_df, on="filename")

    output_file = os.path.join(DATA_PATH, "train_metadata_with_whisper.csv")
    train_meta_with_results.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}") 

if __name__ == "__main__":
    main()
