import os
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Set the environment variables
os.environ['SUNO_ENABLE_MPS'] = "True"          # Enable Apple Silicon Chipset Support
os.environ["SUNO_OFFLOAD_CPU"] = "True"         # 
os.environ["SUNO_USE_SMALL_MODELS"] = "True"    # 

# download and load all models
preload_models()

# Define promts for generations

english_audio_file_name = "Samples/bark_english.wav"
if not os.path.exists(english_audio_file_name):
    text_prompt = """
    Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
    But I also have other interests such as playing tic tac toe.
    """
    audio_array = generate_audio(text_prompt)
    write_wav(english_audio_file_name, SAMPLE_RATE, audio_array)

music_audio_file_name = "Samples/bark_music.wav"
if not os.path.exists(music_audio_file_name):
    text_prompt = """
    ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
    """
    audio_array = generate_audio(text_prompt)
    write_wav(music_audio_file_name, SAMPLE_RATE, audio_array)

v2_en_speaker_1_file_name = "Samples/bark_v2_en_speaker_1.wav"
if not os.path.exists(v2_en_speaker_1_file_name):
    text_prompt = """
    I have a silky smooth voice, and today I will tell you about 
    the exercise regimen of the common sloth.
    """
    audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_1")
    write_wav(v2_en_speaker_1_file_name, SAMPLE_RATE, audio_array)

v2_it_speaker_7_file_name = "Samples/bark_v2_it_speaker_7.wav"
if not os.path.exists(v2_it_speaker_7_file_name):
    text_prompt = """
    ♪ O partigiano porta mi via
    O bella ciao, bella ciao, bella ciao ciao ciao
    O partigiano porta mi via
    Che mi sento di morir ♪
    """
    audio_array = generate_audio(text_prompt, history_prompt="v2/it_speaker_2")
    write_wav(v2_it_speaker_7_file_name, SAMPLE_RATE, audio_array)

v2_ru_speaker_9_file_name = "Samples/bark_v2_ru_speaker_9.wav"
if not os.path.exists(v2_ru_speaker_9_file_name):
    text_prompt = """
    Ванечка, я мячик. Ну и - мне так нравится когда ты меня касаешься...
    """
    audio_array = generate_audio(text_prompt, history_prompt="v2/ru_speaker_9")
    write_wav(v2_ru_speaker_9_file_name, SAMPLE_RATE, audio_array)