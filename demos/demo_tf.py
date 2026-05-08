import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tf_transforms.transforms import Elelet, EleSpectrogram, EleCC

SAMPLE_RATE = 16000
N_FFT = 8192
HOP_LENGTH = 320
NUM_CHANNELS = 128
FMIN = 5
FMAX = 500
AUDIO_PATH = PROJECT_ROOT / "data/test_examples_rumbles/ADDO2012A008.WAV_a0014_10.wav"

audio, sr = librosa.load(AUDIO_PATH)

# resample to 16khz if needed and make mono
if sr != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    sr = SAMPLE_RATE
if len(audio.shape) > 1:
    audio = librosa.to_mono(audio)

audio_torch = torch.tensor(audio).unsqueeze(0)

# audio_torch = torch.zeros_like(audio_torch)
# audio_torch[:,32000] = 1.0

# Test 1: Elelet transform
print("=" * 60)
print("Test 1: Elelet Transform")
print("=" * 60)
elelet_transform = Elelet(
    kernel_size=N_FFT,
    num_channels=NUM_CHANNELS,
    stride=HOP_LENGTH,
    fmin=FMIN,
    fmax=FMAX,
    fs=SAMPLE_RATE,
    supp_mult=1,
    scale='elelog',
    use_torch=True,
)
coeffs = torch.log(torch.abs(elelet_transform(audio_torch.unsqueeze(0)))**2).squeeze(0)
print(f"Elelet output shape: {coeffs.shape}")

print(f"kernels shape: {elelet_transform.kernels.shape}")

# Test 2: EleSpectrogram with elelog scale
print("\n" + "=" * 60)
print("Test 2: EleSpectrogram with elelog scale")
print("=" * 60)

mel_spec = EleSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    fmin=FMIN,
    fmax=FMAX,
    n_mels=NUM_CHANNELS,
    scale='elelog',
    power=1.0,
)
mel_output = mel_spec(audio_torch)
print(f"EleSpectrogram output shape: {mel_output.shape}")
print(f"Min: {mel_output.min().item():.4f}, Max: {mel_output.max().item():.4f}")

# Test 3: EleCC with elelog scale
print("\n" + "=" * 60)
print("Test 3: EleCC with elelog scale")
print("=" * 60)
mfcc_transform = EleCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40,
    log_mels=True,
    melkwargs={
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "fmin": FMIN,
        "fmax": FMAX,
        "n_mels": NUM_CHANNELS,
        "scale": "elelog",
    },
)
mfcc_output = mfcc_transform(audio_torch)
print(f"EleCC output shape: {mfcc_output.shape}")
print(f"Min: {mfcc_output.min().item():.4f}, Max: {mfcc_output.max().item():.4f}")

# Test 4: Compare with standard mel scale
print("\n" + "=" * 60)
print("Test 4: EleSpectrogram with standard mel scale")
print("=" * 60)
mel_spec_standard = EleSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    fmin=FMIN,
    fmax=FMAX,
    n_mels=NUM_CHANNELS,
    scale='mel',
)
mel_output_standard = mel_spec_standard(audio_torch)
print(f"Standard Mel output shape: {mel_output_standard.shape}")
print(f"Min: {mel_output_standard.min().item():.4f}, Max: {mel_output_standard.max().item():.4f}")

# Visualize
print("\n" + "=" * 60)
print("Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Elelet
axes[0, 0].imshow(coeffs.numpy(), aspect='auto', origin='lower', cmap='magma')
axes[0, 0].set_title('Elelet Coefficients')
axes[0, 0].set_ylabel('Elelet Bin')
axes[0, 0].set_xlabel('Time')

# EleSpectrogram (elelog)
axes[0, 1].imshow(torch.log(mel_output[0]).numpy(), aspect='auto', origin='lower', cmap='magma')
axes[0, 1].set_title('EleSpectrogram (elelog scale)')
axes[0, 1].set_ylabel('Elelet Bin')
axes[0, 1].set_xlabel('Time')

# EleSpectrogram (standard mel)
axes[1, 1].imshow(mfcc_output[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[1, 1].set_title('EleCC (elelog scale)')
axes[1, 1].set_ylabel('EleCC Coefficient')
axes[1, 1].set_xlabel('Time')

# EleCC
axes[1, 0].imshow(torch.log(mel_output_standard[0]).numpy(), aspect='auto', origin='lower', cmap='magma')
axes[1, 0].set_title('EleSpectrogram (standard mel scale)')
axes[1, 0].set_ylabel('Mel Bin')
axes[1, 0].set_xlabel('Time')

plt.tight_layout()
plt.show()
