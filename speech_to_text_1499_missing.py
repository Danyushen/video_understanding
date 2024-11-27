import whisper
import os
from moviepy import VideoFileClip
import sys
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

def mp4_to_wav(mp4_file, wav_file):
    video = VideoFileClip(mp4_file)
    video.audio.write_audiofile(wav_file)
    video.close()

def transcribe_audio(model, video, audio_path, sub_path):
    transcript = model.transcribe(
        word_timestamps=True,
        audio=audio_path
    )

    with open(f'{sub_path}/{video[:-4]}_sub.txt', "w") as f:
        for segment in transcript['segments']:
            #sentence-level
            f.write(f"{segment['text']} [{segment['start']}/{segment['end']}]\n")
    
    pass

video_path = "/data1/video_understanding/dataset/video/600-1499-videos"
sub_path = "/data1/video_understanding/dataset/sub/"
audio_path = "/data1/video_understanding/dataset/audio"

model = whisper.load_model("base", device="cuda")

videos = [video for video in os.listdir(video_path) if video.endswith('.mp4')]

for video in tqdm(videos, desc="Processing videos", unit="video"):
    with open(os.devnull, 'w') as devnull:
        if f'{video[:-4]}_sub.txt' in os.listdir(sub_path):
            pass
        else:
            try:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    mp4_to_wav(f'{video_path}/{video}', f'{audio_path}/{video[:-4]}.wav')
                    transcribe_audio(model, video, f'{audio_path}/{video[:-4]}.wav', sub_path)
            except Exception as e:
                print(f"Error processing video {video}: {e}", file=sys.stderr)