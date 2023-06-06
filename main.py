from visualizer import visualizer
import librosa
from generate_data import demux_wav

file = 'D:/Workfile/Project/NSM_ML/NSM_HM/dataset/fan/id_00/normal/00000000.wav'

def test(file):
    sr, y = demux_wav(file)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        S=None,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        power=2.0,
    )

    visualizer.spec_power_plot(S=mel_spectrogram,sr=sr)

test(file)