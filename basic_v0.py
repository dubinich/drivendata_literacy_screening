from loguru import logger
import numpy as np
import pandas as pd
import torch
import whisper
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from transformers import T5ForConditionalGeneration, AutoTokenizer

DATA_PATH = "/mnt/c/Users/artem/Documents/Projects/drivendata_literacy/data"
AUDIO_PATH = DATA_PATH + "/audio"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "facebook/wav2vec2-large-960h"
model_name = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)


def download_whisper_model(download_root="assets"):
    """Code to download model locally so we can include it in our submission"""
    whisper.load_model("turbo", download_root=download_root)

def load_train_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path + "/train_metadata.csv")
    return df

train_meta = load_train_data(DATA_PATH)
sample = train_meta.sample(10, random_state=42)

def run_whisper(data: list[str]):
    model = whisper.load_model("assets/large-v3-turbo.pt").to(DEVICE)
    transcribed_texts = []
    for filename in data:
        result = model.transcribe(AUDIO_PATH+f"/{filename}", language="english", temperature=0.0)
        transcribed_texts.append(result["text"])
    return transcribed_texts


def load_audio(file_path):
    # Load audio file with torchaudio
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz input)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000

    # Convert to mono (single channel)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform.squeeze(0), sample_rate


def preprocess_audio(waveform, sample_rate):
    # Normalize audio to the [-1, 1] range
    waveform = waveform / torch.max(torch.abs(waveform))

    # Tokenize the waveform
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    return inputs.input_values

def transcribe_audio(input_values):
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription


def main():
    train_meta = load_train_data(DATA_PATH)
    sample = train_meta.sample(10, random_state=42)
    txt_model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
    txt_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    sentences = sample["expected_text"].tolist()
    # Tokenize words in each sentence and add prefix
    words_txt = [[f'<eng-us>: {word}' for word in sentence.split()] for sentence in sentences]

    # Flatten the list of words for tokenizer input
    flattened_words = [word for sentence in words_txt for word in sentence]

    # Tokenize with padding
    out = txt_tokenizer(flattened_words, padding=True, add_special_tokens=False, return_tensors='pt')

    # Generate predictions
    preds = txt_model.generate(**out, num_beams=1, max_length=50)

    # Decode predictions
    phonemes_flat = txt_tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)

    # Repack predictions to match the original batch structure
    phonemes_packed = []
    index = 0
    for sentence in words_txt:
        num_words = len(sentence)
        phonemes_packed.append(phonemes_flat[index:index + num_words])
        index += num_words

    # Print results
    for sentence, phonemes in zip(sentences, phonemes_packed):
        print(f"Sentence: {sentence}")
        print(f"Phonemes: {phonemes}")
        print("-"*30)

    # whisper_result = run_whisper(sample["filename"].tolist())
    for filename, expected in zip(sample["filename"].tolist(), sample["expected_text"].tolist()):
        waveform, sample_rate = load_audio(AUDIO_PATH+f"/{filename}")
        input_values = preprocess_audio(waveform, sample_rate)
        transcription = transcribe_audio(input_values)
        print(f"{expected}: {transcription}")


    # for filename, expected_text, result in zip(
    #     sample["filename"].tolist(), sample["expected_text"].tolist(), whisper_result):
    #     print(f"{filename}: {expected_text} -/- {result}")

if __name__ == "__main__":
    main()