import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import torchaudio

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

set_seed(555)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0].unsqueeze(0)

torchaudio.save("output.wav", waveform, sample_rate=model.config.sampling_rate)