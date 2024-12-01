import os
import dl_translate as dlt
import sys
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

sub_path = '/data1/video_understanding/dataset/sub_medium/1-99-med-sub'
sub_path_en = '/data1/video_understanding/dataset/sub_medium/1-99-med-sub-en'

subs = [sub for sub in os.listdir(sub_path)]

mt = dlt.TranslationModel("m2m100", device = "cuda")

sys.stderr = open(f'{sub_path_en}/99_en——error.log', 'w')

for sub in tqdm(subs, desc="Processing subs", unit="sub"):
    with open(os.devnull, 'w') as devnull:
        try:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                for line in open(f'{sub_path}/{sub}'):
                    line = mt.translate(line, source=dlt.lang.CHINESE, target=dlt.lang.ENGLISH)
                    with open(f'{sub_path_en}/{sub[:-4]}_en.txt', 'a') as f:
                        f.write(line + '\n')
        except Exception as e:
            print(f"Error processing video {sub}: {e}", file=sys.stderr)
