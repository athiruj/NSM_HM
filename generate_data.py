# manage & generate data

import sys
import os
import glob

import numpy
import librosa
from logger import logger

from tqdm import tqdm


########################################################################
# Load wav file input
########################################################################


def file_load(wav_name, mono=False):
    """
    load .wav file.

    # Parameters :
    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    # Return :
    : numpy.array( float )
    """
    try:
        y, sr = librosa.load(wav_name, sr=None, mono=mono)
        return y, sr
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################
# Demux wav
########################################################################


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    Enabled to read multiple sampling rates.
    Enabled even one channel.

    # Parameters :
    `wav_name` : str
        target .wav file
    `channel` : int
        target channel number

    # Return :
    numpy.array( float )
        demuxed mono data

    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        logger.warning(f"{msg}")


########################################################################
# File to vector np array
########################################################################


def file_to_vector_array(
    file_name, 
    n_mels=64, 
    frames=5, 
    n_fft=1024, 
    hop_length=512, 
    power=2.0
):
    """
    convert file`[file_name]` to a vector array.

    # Parameters :
    `file_name` : str
        target .wav file

    `n_mels` : int > 0 [scalar]
        number of Mel bands to generate

    `frames` : int
        length of vector size

    `n_fft` : int > 0 [scalar]
        length of the FFT window

    `hop_length` : int > 0 [scalar]
        number of samples between successive frames. See librosa.stft
    
    `power` : float > 0 [scalar] 
        Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc
  
    # Return :
    numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    # ๑ คำนวณมิติของชุดข้อมูล
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    # ๒ คำนวณค่าสเปกโตรแกรม ในความถี่ถูกแปลงเป็นสเกลเมล โดยใช้ Librosa 
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        S=None,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )

    # 03 convert melspectrogram to `log mel energy`(decibel (dB) units) see librosa.power_to_dB
    # ๓ แปลงค่า เมลสเปกโตรแกรม เป็น หน่วยเดซิเบล(dB) โดนใช้ค่า e(epsilon) เป็น อ้างอิง (ref) 
    log_mel_spectrogram = (
        20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    )

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multi_frames
    vector_array = numpy.zeros((vector_array_size, dims), float)

    for t in range(frames):
        vector_array[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[
            :, t : t + vector_array_size
        ].T

    return vector_array


########################################################################
# List to vector np array
########################################################################


def list_to_vector_array(
    file_list, msg="calc...", n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0
):
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(
            file_list[idx],
            n_mels=n_mels,
            frames=frames,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
        )

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[
            vector_array.shape[0] * idx : vector_array.shape[0] * (idx + 1), :
        ] = vector_array

    return dataset


########################################################################
# dataset generator #
########################################################################


def dataset_generator(
    target_dir, normal_dir_name="normal", abnormal_dir_name="abnormal", ext="wav"
):
    """
    generater dataset

    taeget_drr: str
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(
        glob.glob(
            os.path.abspath(
                "{dir}/{normal_dir_name}/*.{ext}".format(
                    dir=target_dir, normal_dir_name=normal_dir_name, ext=ext
                )
            )
        )
    )
    print(normal_files)

    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(
        glob.glob(
            os.path.abspath(
                "{dir}/{abnormal_dir_name}/*.{ext}".format(
                    dir=target_dir, abnormal_dir_name=abnormal_dir_name, ext=ext
                )
            )
        )
    )
    print(abnormal_files)

    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files) :]
    train_labels = normal_labels[len(abnormal_files) :]

    eval_files = numpy.concatenate(
        (normal_files[: len(abnormal_files)], abnormal_files), axis=0
    )
    eval_labels = numpy.concatenate(
        (normal_labels[: len(abnormal_files)], abnormal_labels), axis=0
    )

    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels
