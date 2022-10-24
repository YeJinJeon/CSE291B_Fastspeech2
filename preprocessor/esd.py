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
    gender_dirs = os.listdir(in_dir)[:-1] # except readme.txt
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    for dataset in gender_dirs:
        print("Processing {}ing set...".format(dataset))
        with codecs.open(os.path.join(in_dir, dataset, "{}.txt".format(dataset)), "rb") as f:
            for line in tqdm(f):
                if b"\xa3" in line:
                    line = line.replace(b"\xa3", b",").replace(b"\xac", b" ")
                if b'\xfe' in line or b'\xff' in line:
                    line = line.replace(b"\xfe", b"").replace(b"\xff", b"")
                if b'\x00' in line:
                    line = line.replace(b"\x00", b"")
                line = line.decode("utf-8") # decode bytes to str
                contents = line.strip("\n").strip("\r").split("\t")
                print(contents)
                if not '' in contents and len(contents) == 3:
                    wav_name, text, emotion = contents
                else:
                    continue
                speaker = dataset
                text = _clean_text(text, cleaners)
                wav_path = os.path.join(in_dir, dataset, "*", "*", wav_name+".wav")
                wav_file = glob.glob(wav_path)[0]
                if os.path.exists(wav_file):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_file, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, wav_name+".wav"),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_name)),
                        "w",
                    ) as f1:
                        f1.write(text)
                    with open(
                        os.path.join(out_dir, speaker, "{}.txt".format(wav_name)),
                        "w",
                    ) as f2:
                        f2.write(emotion)