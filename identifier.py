import sys
import re
import time
import vlc
from dejavu import Dejavu
from dejavu.logic.recognizer.microphone_recognizer import MicrophoneRecognizer
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

BG = '/home/antonyxiao/dejavu/backgrounds/'
MIC_LEN = 10 # 10 seconds
PB_OFFSET = 0.45 # playback offset in seconds

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",  # Your MySQL username, 'root' is the default
        "passwd": "debang",  # Your MySQL password
        "db": "dejavu",  # Your database name
    }
}

djv = Dejavu(config)

def format_time(seconds):
    if seconds > 0:
        return '%d:%02d' % (seconds / 60, seconds % 60)
    else:
        return 'N/A'


def print_result(song):
    if 'results' in song:
        for r in song['results']:
            print(str(r['song_name'], encoding='utf-8') + ' ' + format_time(r['offset_seconds']))

    else:
        for r in song[0]:
            print(str(r['song_name'], encoding='utf-8') + ' ' + format_time(r['offset_seconds']))

song = ''

if len(sys.argv) == 1:
    # laptop mic
    # mr = MicrophoneRecognizer(djv, 44100, 1, 18)

    # scarlett solo
    mr = MicrophoneRecognizer(djv, 4096, 44100, 1, 4)
    song = mr.recognize(seconds=MIC_LEN)

elif len(sys.argv) == 2:
    song = djv.recognize(FileRecognizer, sys.argv[1])

print_result(song)


if 'results' in song:
    name = str(song['results'][0]['song_name'], encoding='utf-8')
    play_time = song['results'][0]['offset_seconds']

else:
    name = str(song[0][0]['song_name'], encoding='utf-8')
    play_time = song[0][0]['offset_seconds']

filename = re.findall(r'^\w+?(?=_)|^\w+$', name)
bg_name = filename[0] + '_bg.mp3'

print(bg_name)

bg_url = BG + bg_name

Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new(bg_url)
Media.get_mrl()
player.set_media(Media)
Media.add_option('start-time=' + str(play_time + MIC_LEN + PB_OFFSET))
print(str(round(play_time) + MIC_LEN + PB_OFFSET))
player.play()

time.sleep(30)


# mr = MicrophoneRecognizer(djv, 4096, 44100, 1, 4)
# song = mr.recognize(seconds=10)

