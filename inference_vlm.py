import torch
from PIL import Image, UnidentifiedImageError
import requests, os
from transformers import AutoTokenizer
import librosa
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch

from transformers import CLIPImageProcessor, SiglipImageProcessor
from torch import nn
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
import json
import os
import random
import requests
from PIL import Image
import requests
from io import BytesIO
from PIL import Image
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, Qwen2ForCausalLM
import torch
import pandas as pd
import gzip
from llava.model import LlavaQwen2ForCausalLM, LlavaQwen2Config
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from llava.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, 
                             DEFAULT_IMAGE_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
                             DEFAULT_AUDIO_TOKEN,
                             DEFAULT_AUDIO_START_TOKEN,
                             DEFAULT_AUDIO_END_TOKEN
                             )
MODEL_PATH = "./weights/svla-sft-text-ins"
from inference.tokens_to_audio import decode_speech


speed = 1.0
def get_enhanced_input(prompt_text=">>> ", 
                      history_file=".input_history",
                      completer_words=None):
    """
    Enhanced input function with history, auto-suggestion, and completion.
    
    Args:
        prompt_text (str): Text to show as prompt
        history_file (str): File to store command history
        completer_words (list): List of words for auto-completion
    """
    # Create completer if words are provided
    completer = WordCompleter(completer_words) if completer_words else None
    
    try:
        user_input = prompt(
            prompt_text,
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer
        )
        return user_input
    except KeyboardInterrupt:
        return None
    except EOFError:
        return None

IMAGE_TOKEN_INDEX = -200

system = "System: You serve as a language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def resize_image_if_necessary( image):

        original_width, original_height = image.size
        longest_dimension = 896
        # Determine whether resizing is necessary
        if original_width <= longest_dimension and original_height <= longest_dimension:
            # No resizing needed, return the original image
            return image

        # Determine the scaling factor based on which dimension is larger
        if original_width > original_height:
            # Resize width to the target longest dimension and adjust height proportionally
            new_width = longest_dimension
            new_height = int((longest_dimension / original_width) * original_height)
        else:
            # Resize height to the target longest dimension and adjust width proportionally
            new_height = longest_dimension
            new_width = int((longest_dimension / original_height) * original_width)

        # Resize the image while maintaining aspect ratio
        resized_image = image.resize((new_width, new_height))

        return resized_image

def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True)
    # model = model.bfloat16()
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map="cuda:0")
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor
 
 
def process_image(image, image_processor):
    try:
        processor_output = image_processor.preprocess(image, return_tensors="pt").to("cuda")
        return processor_output
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
 
def generate_text(model, tokenizer, image, prompt, max_new_tokens=1024, temperature=0.7, top_p=1.0, repetition_penalty=1.3):
    try:
        if image != None:
            image=image.unsqueeze(0).float().to("cuda:0")
            
        input_ids = tokenizer([prompt], return_tensors="pt",add_special_tokens=False)["input_ids"]
        input_ids = input_ids.to("cuda:0")
        outputs = model.generate(
            inputs=input_ids,
            images=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.0,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise e
        print(f"Error generating text: {e}\n")
        return None
 
 
 
def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        image.load()  # This will raise an UnidentifiedImageError if the file is not a valid image
        return resize_image_if_necessary(image)
    except requests.RequestException as e:
        print(f"Error loading image: {e}\n")
        return None
    except UnidentifiedImageError:
        print("Error: The URL does not point to a valid image file.\n")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}\n")
        return None
    
def load_image_from_path(path):
    try:
        return resize_image_if_necessary(Image.open(path))
    except Exception as e:
        print(f"Error loading image from file path: {e}\n")
#eos_token_id=[11,2012]
def main():
    model_path = MODEL_PATH
    
    model, tokenizer, image_processor = load_model_and_tokenizer(model_path)
    text_to_audio_model = TTS(language='EN', device="cuda:1")
    speaker_ids = text_to_audio_model.hps.data.spk2id
    speech_output_path = "speech_question.wav"
    asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    
    print("Loading model and tokenizer...\n")
    print("************************************************* IM READY! *************************************************\n")
    while True:
        # input_source = input("Enter the URL or the file path of the image (or 'quit' to exit): ")
        input_source =  get_enhanced_input(prompt_text="Enter the URL or the file path of the image (or 'quit' to exit): ")
        if input_source.lower() == 'quit':
            break

        image = None
        if input_source.startswith('http'):  # Assume it's a URL if it starts with 'http'
            print("Loading image from URL...\n")
            while image is None:
                image = load_image_from_url(input_source)
                if image is None:
                    # input_source = input("Please enter a new URL (or 'quit' to exit): ")
                    input_source = get_enhanced_input(prompt_text="Please enter a new URL (or 'quit' to exit): ")
                    if input_source.lower() == 'quit':
                        break
        elif os.path.exists(input_source):  # Check if it's a valid file path
            print("Loading image from file path...\n")
            image = load_image_from_path(input_source)
            if image is None:
                # input_source = input("Please enter a valid file path (or 'quit' to exit): ")
                input_source = get_enhanced_input(prompt_text="Please enter a new URL (or 'quit' to exit): ")
                if input_source.lower() == 'quit':
                    break
                    
        elif input_source == "no image":
            image == None
            print("you are not using images.")
        else:
            print("Invalid input. Please enter a valid URL or file path.")
            continue

        
        if image:
            image = image_processor(image, return_tensors='pt')["pixel_values"][0]
        round_count =0
        while True:
            # prompt = input("Enter the prompt (or 'end' to switch image, 'quit' to exit): ")
            prompt = get_enhanced_input("Enter the prompt ('audio' to input audio or 'end' to switch image, 'quit' to exit): ")
            if prompt == None or prompt == "":
                print("invalid prompt\n")
                continue
            if prompt.lower() == 'quit':
                return
            if prompt.lower() == 'end':
                break
            
            if prompt.lower() in ['audio', 'speech']:
                while True:
                    textual_speech = get_enhanced_input("Type you speech here ('end' to back to text or  'quit' to exit):")
                    if textual_speech == None or textual_speech == "":
                        print("type you input again")
                        continue
                    elif prompt.lower() == 'quit':
                        return
                    elif prompt.lower() == 'end':
                        break
                    print("saving speech question to 'speech_question.wav'")
                    speaker = random.choice(['EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'])
                    text_to_audio_model.tts_to_file(textual_speech, speaker_ids[speaker], speech_output_path, speed=speed)
                    prompt = audio_encoder(speech_output_path)
                    break
            round_count += 1
            # if (mode == 'single' or round_count == 1) and mode != 'pretrain' and mode != 'cot':
            #     formatted_prompt = f"{system}User:<image>{prompt} Assistant:\n"
            # elif mode == 'pretrain':
            #     formatted_prompt = f'<image>{prompt}'
            system = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
            if image != None:
                formatted_prompt = f"{system}\n<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = f"{system}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            generated_text = generate_text(model, tokenizer, image, formatted_prompt)
            if generated_text:
                print("************************************************* INPUT *************************************************\n")
                print(formatted_prompt)
                print("\n" + "-"*50 + "\n")
                print("************************************************* OUTPUT *************************************************\n")
                print(generated_text)
                print("\n" + "-"*50 + "\n")
                if "||audio-" in generated_text:
                    generated_text = generated_text.replace(".","")
                    print("saving speech answer to 'speech_answer.wav'")
                    decode_speech(generated_text, "cuda:0", "speech_answer.wav")
                    
                    while True:
                        asr = get_enhanced_input("do you want to do ASR for the output speech (y/yes/n/no) : ")
                        if asr.lower() in ['y', 'yes']:
                            print("doing ASR")
                            audio, rate = librosa.load("speech_answer.wav", sr=16000)
                            
                            input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest").input_values
                            logits = asr_model(input_values).logits
                            predicted_ids = torch.argmax(logits, dim=-1)
                            transcription = asr_tokenizer.decode(predicted_ids[0])
                            print(f"ANSWERING IN SPEECH: {transcription}")
                            break
                        break

                            
                    
                    


    print("Thank you for using the configurable image-based conversation generator!\n")

if __name__ == "__main__":
    main()
