from gtts import gTTS
import pygame
import os


def get_audio(text):
    tts = gTTS(text)
    audio_file = f"{text}.mp3"
    tts.save(audio_file)
    sound = pygame.mixer.Sound(audio_file)
    return sound, audio_file


def clean_up_audio(audio_file):
    if os.path.exists(audio_file):
        os.remove(audio_file)
