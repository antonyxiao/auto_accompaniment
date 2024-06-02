import pickle
import sys
import re
import os
import aubio
import numpy as np
import librosa

BLOCK_SIZE = 2048

# Process the audio data in blocks.
def get_pitch(audio_data, block_size):

    pitches = []

    for i in range(0, len(audio_data), block_size):
        block = audio_data[i:i+block_size]
        if len(block) < block_size:
            # If the last block is smaller than block_size, pad it with zeros
            block = np.pad(block, (0, block_size - len(block)), 'constant', constant_values=(0, 0))

        # Aubio's pitch detection.
        pDetection = aubio.pitch("default", 2048, BLOCK_SIZE, 44100)
        pDetection.set_unit("Hz")
        pDetection.set_silence(-40)
        pitch = pDetection(block)[0]
        volume = np.sum(block**2)/len(block)
        volume = "{:.6f}".format(volume)

        pitches.append(pitch)

        # print(pitch)
        # print(volume)

    return pitches

def convert_to_intervals(pitches):
    intervals = []
    for i in range(1, len(pitches)):
        if pitches[i] != 0 and pitches[i-1] != 0:  # Ignore zero pitches (silences)
            interval = pitches[i] - pitches[i-1]
            intervals.append(interval)
    return intervals


audio_dir = 'mp3'

interval_dict = dict()

for file in os.listdir(audio_dir):
    if '.mp3' in file or '.wav' in file or '.flac' in file:
        print('Processing ' + file)
        file_dir = os.path.join(audio_dir, file)
        audio_data, sample_rate = librosa.load(file_dir, sr=44100, mono=True)
        pitches = get_pitch(audio_data, BLOCK_SIZE)
        intervals = convert_to_intervals(pitches)

        file_name_wo_ext = re.findall(r'\w+(?=\.)', file)[0]
        interval_dict[file_name_wo_ext] = intervals


with open('intervals.pkl', 'wb') as f:
    pickle.dump(interval_dict, f)


