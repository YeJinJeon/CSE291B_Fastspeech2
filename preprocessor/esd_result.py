import os
import glob

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import codecs

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    for data in os.listdir(in_dir):
        print(data)
        text = data.split(".")[0]
        wav_file_path = os.path.join(in_dir,data)
        if os.path.exists(wav_file_path):
            print("here")
            os.makedirs(os.path.join(out_dir), exist_ok=True)
            wav, _ = librosa.load(wav_file_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, text+".wav"),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, "{}.lab".format(text)),
                "w",
            ) as f1:
                f1.write(text)
        # for line in tqdm(f):
        #     if b"\xa3" in line:
        #         line = line.replace(b"\xa3", b",").replace(b"\xac", b" ")
        #     if b'\xfe' in line or b'\xff' in line:
        #         line = line.replace(b"\xfe", b"").replace(b"\xff", b"")
        #     if b'\x00' in line:
        #         line = line.replace(b"\x00", b"")
        #     line = line.decode("utf-8") # decode bytes to str
        #     contents = line.strip("\n").strip("\r").split("\t")
        #     if not '' in contents and len(contents) == 3:
        #         wav_name, text, emotion = contents
        #     else:
        #         continue
        #     speaker = dataset
        #     text = _clean_text(text, cleaners)
        #     wav_path = os.path.join(in_dir, dataset, "*", "*", wav_name+".wav")
        #     wav_file = glob.glob(wav_path)[0]