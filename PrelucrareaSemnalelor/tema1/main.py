import numpy as np
import math
import sounddevice
from scipy import signal

rate = 44101
amp = 10000
fullNote = 0.5
note = {}
noteCantec = {}
note['LA4'] = 440
semiton = math.pow(2, 1 / 12)
ton = math.pow(semiton, 2)


def generate_note(octava):
    if octava > 1 and octava != 4:
        note['LA' + str(octava)] = note['LA' + str(octava - 1)] * 2
    elif octava == 1:
        note['LA' + str(octava)] = note['LA4'] / math.pow(2, 4)
    note['DO' + str(octava)] = note['LA' + str(octava)] / semiton / math.pow(ton, 4)
    note['DO#' + str(octava)] = note['DO' + str(octava)] * semiton
    note['RE' + str(octava)] = note['DO' + str(octava)] * ton
    note['RE#' + str(octava)] = note['RE' + str(octava)] * semiton
    note['MI' + str(octava)] = note['RE' + str(octava)] * ton
    note['FA' + str(octava)] = note['MI' + str(octava)] * semiton
    note['FA#' + str(octava)] = note['FA' + str(octava)] * semiton
    note['SOL' + str(octava)] = note['FA' + str(octava)] * ton
    note['SOL#' + str(octava)] = note['SOL' + str(octava)] * semiton
    note['LA#' + str(octava)] = note['LA' + str(octava)] * semiton
    note['SI' + str(octava)] = note['LA' + str(octava)] * ton


for i in range(4, 11):
    generate_note(i)
for i in range(1, 4):
    generate_note(i)
# print(note)


def sinus(frecventa, timp):
    return np.sin(2 * frecventa * timp)


def generate_song(note_cantec):
    signalCantec = []
    for (nota, durata) in note_cantec:
        t = np.linspace(0, fullNote * 1 / durata, int(fullNote * 1 / durata * rate))
        signalCantec.append(amp * sinus(note[nota], t))
    signalComplet = np.concatenate(signalCantec)
    return signalComplet


def play_song(song):
    wav_wave = np.array(song, dtype=np.int16)
    sounddevice.play(wav_wave, blocking=True)
    sounddevice.stop()


noteCantec = [
    ('FA#4', 2), ('MI4', 2),
    ('RE4', 2), ('DO#4', 2),
    ('SI3', 2), ('LA3', 2),
    ('SI3', 2), ('DO#4', 2),
    ('FA#4', 2), ('MI4', 2),
    ('RE4', 2), ('DO#4', 2),
    ('SI3', 2), ('LA3', 2)
]


noteCraciun = []
f = open("note_muzicale.txt")
input = f.read()
x = np.array(input.split("\n"))

for pair in x:
    elements = pair.split(" ")
    noteCraciun.append([elements[0], float(elements[1])])
# print(noteCraciun)

play_song(generate_song(noteCraciun))
