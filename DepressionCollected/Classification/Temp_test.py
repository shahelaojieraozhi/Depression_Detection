import numpy as np
import os

prefix = os.path.abspath(os.path.join(os.getcwd(), "."))

audio_features_path = os.path.join(
    prefix, "Features/AudioWhole/whole_samples_reg_256.npz"
)

audio_features = np.load(audio_features_path)["arr_0"]
audio_features = np.squeeze(audio_features, axis=2)
print()

# audio_features = np.squeeze(
#     np.load(audio_features_path)["arr_0"],
#     axis=2,
# )
audio_targets = np.load(
    os.path.join(prefix, "Features/AudioWhole/whole_labels_clf_256.npz")
)["arr_0"]
