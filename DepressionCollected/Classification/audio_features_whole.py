import os
import sys
import wave

import librosa
import numpy as np

import tensorflow.compat.v1 as tf
# import tensorflow as tf
sys.path.append("../Classification")

import loupe_keras as lpk
import warnings

warnings.filterwarnings("ignore")

# tf.enable_eager_execution()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

cluster_size = 16

min_len = 100
max_len = -1


def wav2vlad(wave_data, sr):
    global cluster_size
    signal = wave_data

    melspec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=80)
    melspec = np.log(np.maximum(1e-6, melspec))  # 进行对数变换，避免 log(0) 错误
    melspec = melspec.astype(np.float32).T  # 转置并转换数据类型

    feature_size = melspec.shape[1]
    max_samples = melspec.shape[0]
    output_dim = cluster_size * 16
    feat = lpk.NetVLAD(
        feature_size=feature_size,
        max_samples=max_samples,
        cluster_size=cluster_size,
        output_dim=output_dim,
    )(
        tf.convert_to_tensor(melspec)
    )  # [1, 256]

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        r = feat.numpy()

    return r
    # return melspec


def extract_features(number, audio_features, targets, path):
    global max_len, min_len

    positive_wav_path = os.path.join(
        prefix, "{1}_{0}/positive_out.wav".format(number, path)
    )
    if not os.path.exists(positive_wav_path):
        return
    positive_file = wave.open(positive_wav_path)
    sr1 = positive_file.getframerate()
    nframes1 = positive_file.getnframes()
    wave_data1 = np.frombuffer(
        positive_file.readframes(nframes1), dtype=np.short
    ).astype(np.float32)
    len1 = nframes1 / sr1

    positive_wav_path = os.path.join(
        prefix, "{1}_{0}/neutral_out.wav".format(number, path)
    )
    neutral_file = wave.open(positive_wav_path)
    sr2 = neutral_file.getframerate()
    nframes2 = neutral_file.getnframes()
    wave_data2 = np.frombuffer(
        neutral_file.readframes(nframes2), dtype=np.short
    ).astype(np.float32)
    len2 = nframes2 / sr2

    negative_wav_path = os.path.join(
        prefix, "{1}_{0}/negative_out.wav".format(number, path)
    )
    negative_file = wave.open(negative_wav_path)
    sr3 = negative_file.getframerate()
    nframes3 = negative_file.getnframes()
    wave_data3 = np.frombuffer(
        negative_file.readframes(nframes3), dtype=np.short
    ).astype(np.float32)
    len3 = nframes3 / sr3

    for l in [len1, len2, len3]:
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l

    with open(
        os.path.join(prefix, "{1}_{0}/new_label.txt".format(number, path))
    ) as fli:
        target = float(fli.readline())

    if wave_data1.shape[0] < 1:
        wave_data1 = np.array([1e-4] * sr1 * 5)
    if wave_data2.shape[0] < 1:
        wave_data2 = np.array([1e-4] * sr2 * 5)
    if wave_data3.shape[0] < 1:
        wave_data3 = np.array([1e-4] * sr3 * 5)
    audio_features.append(
        [
            wav2vlad(wave_data1, sr1),
            wav2vlad(wave_data2, sr2),
            wav2vlad(wave_data3, sr3),
        ]
    )
    # targets.append(1 if target >= 53 else 0)
    targets.append(target)


audio_features = []
audio_targets = []

for index in range(114):
    extract_features(index + 1, audio_features, audio_targets, "EATD-Corpus/t")

for index in range(114):
    extract_features(index + 1, audio_features, audio_targets, "EATD-Corpus/v")


print("Saving npz file locally...")

os.makedirs("Features/AudioWhole", exist_ok=True)
np.savez(
    os.path.join(
        prefix, "Features/AudioWhole/whole_samples_reg_%d.npz" % (cluster_size * 16)
    ),
    audio_features,
)
np.savez(
    os.path.join(prefix, "Features/AudioWhole/whole_labels_reg_%d.npz")
    % (cluster_size * 16),
    audio_targets,
)

print(max_len, min_len)
