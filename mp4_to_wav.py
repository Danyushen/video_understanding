import os
from moviepy import VideoFileClip
import sys
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

def mp4_to_wav(mp4_file, wav_file):
    video = VideoFileClip(mp4_file)
    video.audio.write_audiofile(wav_file)
    video.close()

video_path = "/data1/video_understanding/3000-3999-videos"
audio_path = "/data1/video_understanding/dataset/audio/3000-3999-audio"
sub_path = "/data1/video_understanding/dataset/sub"

videos = [video for video in os.listdir(video_path) if video.endswith('.mp4')]

for video in tqdm(videos, desc="Processing videos", unit="video"):
    with open(os.devnull, 'w') as devnull:
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                mp4_to_wav(f'{video_path}/{video}', f'{audio_path}/{video[:-4]}.wav')
        except Exception as e:
            print(f"Error processing video {video}: {e}", file=sys.stderr)