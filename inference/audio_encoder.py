import json
import torchaudio
import torch
from tqdm import tqdm
from speechtokenizer import SpeechTokenizer
import torchaudio
# Initialize the tokenizer and other variables
audio_tokenizer = SpeechTokenizer.load_from_checkpoint(
    './weights/config.json', 
    './weights/ckpt.dev')
audio_sample_rate = audio_tokenizer.sample_rate
device = "cuda:1"
audio_tokenizer = audio_tokenizer.to(device)
def audio_2_text(audio_tokens):
    assert len(audio_tokens.size()) == 1
    numpy_array = audio_tokens.cpu().numpy()
    list_from_tensor = numpy_array.tolist()
    audio_tokens = [f"<||audio-{audio_idx}||>" for audio_idx in list_from_tensor]
    return audio_tokens

def audio_encoder(wav_path):
    wav, audio_sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav[:1, ]
        
    if audio_sr != audio_sample_rate:
            wav = torchaudio.functional.resample(wav, audio_sr, audio_sample_rate)

    wav = wav.unsqueeze(0)
    wav = wav.to(device)
    
    # Move the tokenizer model to the correct device
      # Ensure the tokenizer is on the same GPU
    
    with torch.no_grad():
        wav = audio_tokenizer.encode(wav)
    
    wav = wav[0, 0, :]
    audio_tokens = audio_2_text(wav)
    audio_tokens = "".join(audio_tokens)
    return "<||audio_start||>" + audio_tokens +"<||audio_end||>"