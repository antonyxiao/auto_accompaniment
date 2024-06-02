import aubio
import numpy as num
import pyaudio
import wave

BLOCK_SIZE = 2048

# PyAudio object.
p = pyaudio.PyAudio()

# Open stream.
# stream = p.open(format=pyaudio.paFloat32,
#     channels=1, rate=44100, input=True,
#     frames_per_buffer=4096,
#     input_device_index=4)

stream = p.open(format=pyaudio.paFloat32,
    channels=1, rate=44100, input=True,
    frames_per_buffer=BLOCK_SIZE,
    input_device_index=18)

# Aubio's pitch detection.
pDetection = aubio.pitch("default", 2048, BLOCK_SIZE, 44100)

# Set unit.
pDetection.set_unit("Hz")
pDetection.set_silence(-40)

pitches = []

# Calculate the number of iterations for 10 seconds
total_seconds = 8
samples_per_second = 44100
buffer_size = BLOCK_SIZE
iterations_needed = (total_seconds * samples_per_second) / buffer_size

current_iteration = 0

while current_iteration < iterations_needed:

    data = stream.read(BLOCK_SIZE)
    samples = num.frombuffer(data, dtype=aubio.float_type)
    pitch = pDetection(samples)[0]
    # Compute the energy (volume) of the
    # current frame.
    volume = num.sum(samples**2)/len(samples)
    # Format the volume output so that at most
    # it has six decimal numbers.
    volume = "{:.6f}".format(volume)

    pitches.append(pitch)

    # print(pitch)
    # print(volume)

    current_iteration += 1


def convert_to_intervals(pitches):
    intervals = []
    for i in range(1, len(pitches)):
        if pitches[i] != 0 and pitches[i-1] != 0:  # Ignore zero pitches (silences)
            interval = pitches[i] - pitches[i-1]
            intervals.append(interval)
    return intervals

intervals = convert_to_intervals(pitches)

print(intervals)
