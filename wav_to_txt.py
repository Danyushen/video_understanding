import whisper
import os
import sys
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

def transcribe_audio(model, audio, audio_path, sub_path):
    transcript = model.transcribe(
        word_timestamps=True,
        audio=audio_path
    )

    with open(f'{sub_path}/{audio[:-4]}_sub.txt', "w") as f:
        for segment in transcript['segments']:
            #sentence-level
            f.write(f"{segment['text']} [{segment['start']}/{segment['end']}]\n")
    
    pass

audio_path = "/data1/video_understanding/dataset/audio/2368-2999-audio"
sub_path = "/data1/video_understanding/dataset/sub"

model = whisper.load_model("base", device="cuda")

audios = [video for video in os.listdir(audio_path) if video.endswith('.wav')]

for audio in tqdm(audios, desc="Processing videos", unit="video"):
    with open(os.devnull, 'w') as devnull:
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                transcribe_audio(model, audio, f'{audio_path}/{audio[:-4]}.wav', sub_path)
        except Exception as e:
            print(f"Error processing audio {audio}: {e}", file=sys.stderr)