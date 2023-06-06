from visualizer import visualizer_func
import librosa
from generate_data import demux_wav
import numpy
import sys

file = "D:/Workfile/Project/NSM_ML/NSM_HM/dataset/fan/id_00/normal/00000000.wav"


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

    log_mel_spectrogram = (
        20.0 / 2.0 * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    )

    visualizer = visualizer_func()
    visualizer.spec_power_plot(log_mel=log_mel_spectrogram, sr=sr)
    visualizer.show()
    # visualizer.save_figure("D:/Workfile/Project/NSM_ML/NSM_HM/gg")


test(file)
