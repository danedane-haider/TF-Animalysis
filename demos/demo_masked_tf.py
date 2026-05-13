import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tf_transforms.transforms import EleSpectrogram, MaskedEleSpectrogram, EleCC, MaskedEleCC
from tf_transforms.utils_harmonic_mask import load_f0_csv

SAMPLE_RATE = 16000
AUDIO_PATH = PROJECT_ROOT / "data/test_rumbles/ADDO2012A008.WAV_a0014_10.wav"
F0_PATH = PROJECT_ROOT / "data/test_rumbles/f0_refined/ADDO2012A008.WAV_a0014_10.f0.csv"

audio, _ = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
audio_torch = torch.tensor(audio).unsqueeze(0)

transform_kwargs = dict(
    sample_rate=SAMPLE_RATE,
    n_fft=8192,
    hop_length=320,
    fmin=5,
    fmax=500,
    n_mels=128,
    scale="elelog",
    power=1.0,
)

regular_transform = EleSpectrogram(**transform_kwargs)
masked_transform = MaskedEleSpectrogram(**transform_kwargs)

# Drop the 0.5 factor if your CSV already stores f0 rather than F1.
f0_hz = load_f0_csv(F0_PATH) * 0.5
regular_spec = regular_transform(audio_torch)
masked_spec = masked_transform(
    audio_torch,
    f0_hz,
    width_bins=1,
    n_harmonics=32,
)

print(f"EleSpectrogram output shape: {regular_spec.shape}")
print(f"MaskedEleSpectrogram output shape: {masked_spec.shape}")
print(f"Min: {masked_spec.min().item():.4f}, Max: {masked_spec.max().item():.4f}")

regular_image = torch.log(regular_spec[0] + 1e-10).detach().numpy()
masked_image = torch.log(masked_spec[0] + 1e-10).detach().numpy()
vmin = regular_image.min()
vmax = regular_image.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

axes[0].imshow(regular_image, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
axes[0].set_title("EleSpectrogram")
axes[0].set_ylabel("Elelog Bin")
axes[0].set_xlabel("Time")

axes[1].imshow(masked_image, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
axes[1].set_title("MaskedEleSpectrogram")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.show()


transform_kwargs = dict(
    sample_rate=16000,
    n_mfcc=40,
    log_mels=True,
    melkwargs={
        "n_fft": 8192,
        "hop_length": 320,
        "fmin": 5,
        "fmax": 500,
        "n_mels": 128,
        "scale": "elelog",
    },
)

regular_transform = EleCC(**transform_kwargs)
masked_transform = MaskedEleCC(**transform_kwargs)

# Drop the 0.5 factor if your CSV already stores f0 rather than F1.
f0_hz = load_f0_csv(F0_PATH) * 0.5
regular_spec = regular_transform(audio_torch)
masked_spec = masked_transform(
    audio_torch,
    f0_hz,
    width_bins=1,
    n_harmonics=32,
)

print(f"EleCC output shape: {regular_spec.shape}")
print(f"MaskedEleCC output shape: {masked_spec.shape}")
print(f"Min: {masked_spec.min().item():.4f}, Max: {masked_spec.max().item():.4f}")

regular_image = regular_spec[0].detach().numpy()
masked_image = masked_spec[0].detach().numpy()
vmin = regular_image.min()
vmax = regular_image.max()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

axes[0].imshow(regular_image, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
axes[0].set_title("EleCC")
axes[0].set_ylabel("Cepstral Coefficient")
axes[0].set_xlabel("Time")

axes[1].imshow(masked_image, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
axes[1].set_title("MaskedEleCC")
axes[1].set_xlabel("Time")

plt.tight_layout()
plt.show()