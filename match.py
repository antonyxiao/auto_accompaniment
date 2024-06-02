import pickle
import sys
import numpy as np
import os
import aubio
import librosa
import aubio
import pyaudio
import wave
from dtaidistance import dtw

BLOCK_SIZE = 2048
MIC_SECOND = 10

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


def simple_sequence_matching(short_intervals, long_intervals):
    short_length = len(short_intervals)
    min_difference = float('inf')
    best_match_position = -1
    
    # Iterate over each possible starting position for a subsequence in the long_intervals
    for i in range(len(long_intervals) - short_length + 1):
        difference = sum(abs(short_intervals[j] - long_intervals[i + j]) for j in range(short_length))
        
        if difference < min_difference:
            min_difference = difference
            best_match_position = i
    
    return best_match_position, min_difference

def dtw_sequence_matching(short_intervals, long_intervals):
    short_length = len(short_intervals)
    best_distance = float('inf')
    best_match_position = -1
    
    # Convert lists to numpy arrays for dtaidistance
    short_intervals_np = np.array(short_intervals).reshape(-1,1)
    long_intervals_np = np.array(long_intervals).reshape(-1,1)
    
    # Iterate over each possible subsequence
    for i in range(len(long_intervals) - short_length + 1):
        current_sequence = long_intervals_np[i:i+short_length]
        distance = dtw.distance(short_intervals_np, current_sequence)
        
        if distance < best_distance:
            best_distance = distance
            best_match_position = i
            
    return best_match_position, best_distance


def get_pitches_from_mic():

    # PyAudio object.
    p = pyaudio.PyAudio()

    # Open stream.
    stream = p.open(format=pyaudio.paFloat32,
        channels=1, rate=44100, input=True,
        frames_per_buffer=4096,
        input_device_index=4)
    
    # laptop mic
    # stream = p.open(format=pyaudio.paFloat32,
    #     channels=1, rate=44100, input=True,
    #     frames_per_buffer=BLOCK_SIZE,
    #     input_device_index=18)

    # Aubio's pitch detection.
    pDetection = aubio.pitch("default", 2048, BLOCK_SIZE, 44100)

    # Set unit.
    pDetection.set_unit("Hz")
    pDetection.set_silence(-40)

    pitches = []

    # Calculate the number of iterations for 10 seconds
    total_seconds = MIC_SECOND
    samples_per_second = 44100
    buffer_size = BLOCK_SIZE
    iterations_needed = (total_seconds * samples_per_second) / buffer_size

    current_iteration = 0

    while current_iteration < iterations_needed:

        data = stream.read(BLOCK_SIZE)
        samples = np.frombuffer(data, dtype=aubio.float_type)
        pitch = pDetection(samples)[0]
        # Compute the energy (volume) of the
        # current frame.
        volume = np.sum(samples**2)/len(samples)
        # Format the volume output so that at most
        # it has six decimal numbers.
        volume = "{:.6f}".format(volume)

        pitches.append(pitch)

        # print(pitch)
        # print(volume)

        current_iteration += 1

    return pitches


pitches = []


if len(sys.argv) == 1:
    pitches = get_pitches_from_mic()

elif len(sys.argv) == 2:
    audio_data, sample_rate = librosa.load(sys.argv[1], sr=44100, mono=True)
    pitches = get_pitch(audio_data, BLOCK_SIZE)

input_intervals = convert_to_intervals(pitches)

# database
intervals_dict = dict()

with open('intervals.pkl', 'rb') as file:
    intervals_dict = pickle.load(file)

similarities = dict()

for key in intervals_dict:
    match_position, similarity = simple_sequence_matching(input_intervals, intervals_dict[key])
    similarities[key] = [match_position, similarity]


# Initialize variables to hold the minimum similarity and corresponding key and position
min_similarity = float('inf')
min_key = None
min_position = None

# Iterate over the dictionary
for key, value in similarities.items():
    position, similarity = value
    # Check if the current similarity is lower than the minimum similarity
    if similarity < min_similarity:
        min_similarity = similarity
        min_key = key
        min_position = position


print()

for key in similarities:
    print(key + ': ' + str(similarities[key]))

print('\nBest Match: ')
# Print the result
print(f"{min_key}, Position: {min_position}, Similarity: {min_similarity}")


# to convert freq to midi
#
# midi_pitch = []

# for p in pitches:
#     if p == 0:
#         midi_pitch.append(0)
#     else:
#         midi_pitch.append(round(librosa.hz_to_midi(p)))

# print(midi_pitch)



# intervals = convert_to_intervals(pitches)
# print(intervals)

# import aubio
# import librosa
# import numpy as np

# # Initialize the pitch detection object
# samplerate = 44100  # the sampling rate (samples per second)
# hop_size = 512  # number of frames to analyze at a time
# pitch_o = aubio.pitch("default", 2048, hop_size, samplerate)
# pitch_o.set_unit("Hz")  # Set the unit of the pitch detection to Hertz
# pitch_o.set_tolerance(0.7)

# # Function to extract pitches from an audio file
# def extract_pitch(file_path, samplerate=44100):
#     source = aubio.source(file_path, samplerate, hop_size)
#     pitches = []
#     confidences = []

#     while True:
#         samples, read = source()
#         pitch = pitch_o(samples)[0]
#         confidence = pitch_o.get_confidence()

#         if confidence > 0.7:  # You can adjust this threshold based on your needs
#             pitches.append(pitch)
#             confidences.append(confidence)

#         if read < hop_size:
#             break

#     return np.array(pitches), np.array(confidences)

# Example: Extracting pitches from a file
# file_path = 'mp3/xingqing_vocal.flac'

# pitches, confidences = extract_pitch(file_path)

