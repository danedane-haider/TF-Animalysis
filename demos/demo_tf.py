import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tf_transforms.transforms import Elelet, EleSpectrogram, EleCC
from torchaudio.transforms import MelSpectrogram, MFCC, Spectrogram

SAMPLE_RATE = 16000
N_FFT = 8192
HOP_LENGTH = 320
NUM_CHANNELS = 128
FMIN = 5
FMAX = 500
AUDIO_PATH = PROJECT_ROOT / "data/test_rumbles/ADDO2012A008.WAV_a0014_10.wav"

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
    pad_mode="reflect",
)
elelet_output = torch.log(torch.abs(elelet_transform(audio_torch))**1)
print(f"Elelet output shape: {elelet_output.shape}")

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
mel_output = torch.log(mel_spec(audio_torch))
print(f"EleSpectrogram output shape: {mel_output.shape}")

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

# Test 4: Compare with standard mel scale
print("\n" + "=" * 60)
print("Test 4: MelSpectrogram")
print("=" * 60)
# mel_spec_standard = EleSpectrogram(
#     sample_rate=SAMPLE_RATE,
#     n_fft=N_FFT,
#     hop_length=HOP_LENGTH,
#     fmin=FMIN,
#     fmax=FMAX,
#     n_mels=NUM_CHANNELS,
#     scale='mel',
# )
# mel_output_standard = torch.log(mel_spec_standard(audio_torch))
# print(f"Standard Mel output shape: {mel_output_standard.shape}")
mel_spec_standard = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    f_min=FMIN,
    f_max=FMAX,
    n_mels=NUM_CHANNELS,
)
mel_output_standard = torch.log(mel_spec_standard(audio_torch))
print(f"Standard Mel output shape: {mel_output_standard.shape}")

# Test 5: Compare with standard MFCC
print("\n" + "=" * 60)
print("Test 5: MFCC")
print("=" * 60)
mfcc_transform_standard = MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40,
    log_mels=True,
    melkwargs={
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "f_min": FMIN,
        "f_max": FMAX,
        "n_mels": NUM_CHANNELS,
    },
)
mfcc_output_standard = mfcc_transform_standard(audio_torch)
print(f"Standard MFCC output shape: {mfcc_output_standard.shape}")

# Test 6: Compare with standard spectrogram
print("\n" + "=" * 60)
print("Test 6: Spectrogram")
print("=" * 60)
spectrogram_transform = Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=1.0,
)
spectrogram_output = torch.log(spectrogram_transform(audio_torch))
spectrogram_output = spectrogram_output[:, :FMAX, :]
print(f"Spectrogram output shape: {spectrogram_output.shape}")




# Visualize
print("\n" + "=" * 60)
print("Visualizations")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# figure title
fig.suptitle('Spectrogram vs MelSpectrogram vs EleSpectrogram', fontsize=16)

# Spectrogram
axes[0].imshow(spectrogram_output[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[0].set_title('Spectrogram')
axes[0].set_ylabel('Hz')
axes[0].set_xlabel('Time')

# MelSpectrogram
axes[1].imshow(mel_output_standard[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[1].set_title('MelSpectrogram')
axes[1].set_ylabel('Mel Bin')
axes[1].set_xlabel('Time')

# EleSpectrogram
axes[2].imshow(mel_output[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[2].set_title('EleSpectrogram')
axes[2].set_ylabel('EleScale Bin')
axes[2].set_xlabel('Time')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

fig.suptitle('EleSpectrogram vs Elelet Coefficients', fontsize=16)

# EleSpectrogram
axes[0].imshow(mel_output[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[0].set_title('EleSpectrogram')
axes[0].set_ylabel('EleScale Bin')
axes[0].set_xlabel('Time')

# Elelet
axes[1].imshow(elelet_output[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[1].set_title('Elelet Coefficients')
axes[1].set_ylabel('EleScale Bin')
axes[1].set_xlabel('Time')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

fig.suptitle('MFCC vs EleCC', fontsize=16)

# MFCC
axes[0].imshow(mfcc_output_standard[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[0].set_title('MFCC')
axes[0].set_ylabel('MFCC Bin')
axes[0].set_xlabel('Time')

# EleCC
axes[1].imshow(mfcc_output[0].numpy(), aspect='auto', origin='lower', cmap='magma')
axes[1].set_title('EleCC')
axes[1].set_ylabel('EleCC Bin')
axes[1].set_xlabel('Time')

plt.tight_layout()
plt.show()
