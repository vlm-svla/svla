import torchaudio
import re
import torch
from inference.audio_decoder import load_soundstorm, semantic2acoustic
import json
from speechtokenizer import SpeechTokenizer
device = "cuda:0"
speech_tokenizer = SpeechTokenizer.load_from_checkpoint("./weights/config.json", "./weights/ckpt.dev")     
speech_tokenizer.eval()

soundstorm = load_soundstorm("./weights/speechtokenizer_soundstorm_mls.pt")
soundstorm.eval()


def decode_speech(content, device, prompt_path="checking.wav"):
    speech_tokenizer.to(device=device)
    soundstorm.to(device=device)
    prompt_tokens = None

    # semantic_codes = [[int(num) for num in re.findall(r'\d+', content)]]
    semantic_codes = content.split("||><||")
    semantic_codes = [item.split("-")[-1].replace("||>", "").replace("<||", "") for item in semantic_codes]
    semantic_codes = [int(item) for item in semantic_codes]
    semantic_codes = [semantic_codes]
    # wav: (b, 1, t)
    config_dict = {
    "temperature":0.7,
    "top_p": 0.7,
    "do_sample": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "repetition_penalty": 1.15,
    "vc_steps": 1,
    "num_beams": 1
}
    wav = semantic2acoustic(torch.Tensor(semantic_codes).int().to(device), prompt_tokens, 
                            soundstorm, speech_tokenizer, steps=config_dict['vc_steps'])
    wav = wav.squeeze(0).detach().cpu()
    torchaudio.save(prompt_path, wav, speech_tokenizer.sample_rate)