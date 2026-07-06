# TF-Animalysis Time-Frequency Representations

This repository contains time-frequency transforms for low-frequency animal
vocalizations, with a focus on elephant rumble analysis. The core transforms
live in `tf_transforms/transforms.py` and are demonstrated in `demos/`.

The main idea is to compare ordinary STFT/mel-style representations with
elephant-oriented frequency scales and optional f0-guided harmonic masking.

## Representations

| Class | Output | What it does |
| --- | --- | --- |
| `Elelet` | complex filterbank coefficients | Direct FFT convolution with elephant auditory filters. |
| `EleSpectrogram` | real spectrogram-like tensor | STFT magnitude projected onto an auditory filterbank such as `elelog`. |
| `MaskedEleSpectrogram` | real spectrogram-like tensor | Same as `EleSpectrogram`, but the STFT is first masked around f0 harmonics. |
| `EleCC` | cepstral coefficients | DCT of a log `EleSpectrogram`, analogous to MFCC. |
| `MaskedEleCC` | cepstral coefficients | DCT of a log `MaskedEleSpectrogram`. |

All torch transforms expect audio shaped as:

```python
(batch, time)
```

For one mono waveform:

```python
audio_torch = torch.tensor(audio).unsqueeze(0)
```

## Quick Start

Install dependencies with `uv`:

```bash
uv sync
```

Run the basic transform comparison:

```bash
uv run demos/demo_tf.py
```

Run the masked transform comparison:

```bash
uv run demos/demo_masked_tf.py
```

The demos compare STFT, mel, EleSpectrogram, Elelet, MFCC, EleCC, and the
masked variants on an example rumble.

## F0 extraction workflow

Run all commands from the repository root. The supported extraction workflow
currently uses the `hybrid_elelet` tracker.

### 1. Resample the audio

Convert the source audio to mono, 16 kHz WAV files in a separate working
directory:

```bash
uv run python utils/resample.py \
  --input /path/to/raw_audio \
  --output data/rumbles \
  --sr 16000
```

### 2. Extract the contours

```bash
uv run python f0_extraction/extract_f0.py \
  --input data/rumbles \
  --method hybrid_elelet
```

The extractor writes one CSV per WAV to `data/rumbles/f0_hybrid_elelet/` and
caches the Elelet representations beside the audio for reuse. The CSV
`frequency` column follows the project's H1 convention; the corresponding
DDSP F0 is `H1 / 2`.

### 3. Annotate and correct the contours

```bash
uv run python f0_extraction/annotate_f0.py \
  --input data/rumbles \
  --f0_dir f0_hybrid_elelet \
  --initial_spec_mode elelet
```

Left-click to add correction points, right-click to remove them, and use `W`
and `E` to set the voiced region's start and end. Navigate with the arrow keys
(or `N`/`P`) and press `Q` to save and quit. Corrected contours are written to
`data/rumbles/f0_corrected/` without changing the extracted source contours.
See [the detailed F0/F1 workflow](f0_extraction/README.md) for all annotation
controls and file conventions.

## EleSpectrogram

`EleSpectrogram` is the closest analogue to
`torchaudio.transforms.MelSpectrogram`.

It computes:

```text
audio -> STFT -> magnitude/power -> auditory filterbank
```

Example:

```python
import torch
from tf_transforms.transforms import EleSpectrogram

transform = EleSpectrogram(
    sample_rate=16000,
    n_fft=8192,
    hop_length=320,
    f_min=5,
    f_max=500,
    n_mels=128,
    scale="elelog",
    power=1.0,
)

spec = transform(audio_torch)
print(spec.shape)  # (batch, n_mels, frames)
```

Important parameters:

- `sample_rate`: audio sample rate, usually `16000`.
- `n_fft`: STFT size.
- `hop_length`: frame spacing in samples.
- `f_min`, `f_max`: frequency range in Hz.
- `n_mels`: number of output frequency bins.
- `scale`: auditory scale, for example `"elelog"` or `"mel"`.
- `pad_mode`: STFT boundary padding; default is `"reflect"`.

## Elelet

`Elelet` directly convolves the signal with elephant-oriented filters. Unlike
`EleSpectrogram`, it does not start from an STFT. It returns complex
coefficients, so plotting usually uses magnitude or log magnitude.

```python
from tf_transforms.transforms import Elelet

transform = Elelet(
    kernel_size=8192,
    num_channels=128,
    stride=320,
    f_min=5,
    f_max=500,
    fs=16000,
    supp_mult=1,
    scale="elelog",
    use_torch=True,
    pad_mode="reflect",
    backend="fft_decimated",
    channel_batch_size=64,
)

coeffs = transform(audio_torch)
image = torch.log(torch.abs(coeffs) + 1e-10)
print(coeffs.shape)  # (batch, channels, frames)
```

`pad_mode="circular"` preserves the old wraparound convolution behavior.
`pad_mode="reflect"` mirrors the signal at the boundaries before FFT
convolution, similar in spirit to `torch.stft(center=True, pad_mode="reflect")`.

`backend="fft_decimated"` is the default. It uses spectral aliasing to compute
only the samples selected by `stride`, while preserving the complex Elelet
coefficients. The inverse-FFT reduction is
`gcd(convolution_length, stride)`, so fixed ML input lengths aligned to the
stride receive the largest speedup. Use `backend="fft"` for the full-IFFT
reference implementation.

Kernel spectra are cached for the most recently used input length. This avoids
re-transforming the filters across fixed-size ML batches. Set
`cache_kernel_fft=False` to disable the cache, or call `clear_fft_cache()` after
changing the kernels. `channel_batch_size` bounds intermediate memory without
changing the result; omit it to process every channel together.

## MaskedEleSpectrogram

`MaskedEleSpectrogram` keeps only energy near an f0 contour and its harmonics
before applying the auditory filterbank.

It computes:

```text
audio -> STFT -> sparse harmonic mask -> reproducing-kernel smoothing
      -> masked magnitude -> auditory filterbank
```

The f0 contour should be in Hz. It can be a tensor shaped `(frames,)` or
`(batch, frames)`. If the number of f0 frames differs from the STFT frame count,
the contour is linearly interpolated.

```python
from tf_transforms.transforms import MaskedEleSpectrogram
from tf_transforms.utils_harmonic_mask import load_f0_csv

f0_hz = load_f0_csv("path/to/example.f0.csv")

masked_transform = MaskedEleSpectrogram(
    sample_rate=16000,
    n_fft=8192,
    hop_length=320,
    f_min=5,
    f_max=500,
    n_mels=128,
    scale="elelog",
    power=1.0,
)

masked_spec = masked_transform(
    audio_torch,
    f0_hz,
    width_bins=1,
    n_harmonics=32,
)
```

If your CSV stores F1 rather than f0, divide by two before passing it in:

```python
f0_hz = load_f0_csv(F0_PATH) * 0.5
```

Mask parameters:

- `width_bins`: how many STFT bins to keep around each harmonic.
- `n_harmonics`: number of harmonics to include.
- `kernel_floor`: threshold used when building the reproducing kernel smoother.

To inspect the smoothed mask:

```python
masked_spec, mask = masked_transform(audio_torch, f0_hz, return_mask=True)
```

## EleCC and MaskedEleCC

`EleCC` is the cepstral version of `EleSpectrogram`. It computes:

```text
audio -> EleSpectrogram -> log -> DCT
```

Example:

```python
from tf_transforms.transforms import EleCC

transform = EleCC(
    sample_rate=16000,
    n_mfcc=40,
    log_mels=True,
    melkwargs={
        "n_fft": 8192,
        "hop_length": 320,
        "f_min": 5,
        "f_max": 500,
        "n_mels": 128,
        "scale": "elelog",
    },
)

coeffs = transform(audio_torch)
print(coeffs.shape)  # (batch, n_mfcc, frames)
```

`MaskedEleCC` does the same thing, but starts from `MaskedEleSpectrogram`:

```python
from tf_transforms.transforms import MaskedEleCC

masked_transform = MaskedEleCC(
    sample_rate=16000,
    n_mfcc=40,
    log_mels=True,
    melkwargs={
        "n_fft": 8192,
        "hop_length": 320,
        "f_min": 5,
        "f_max": 500,
        "n_mels": 128,
        "scale": "elelog",
    },
)

masked_coeffs = masked_transform(
    audio_torch,
    f0_hz,
    width_bins=1,
    n_harmonics=32,
)
```

Note that `n_mfcc`, `log_mels`, and `melkwargs` are constructor arguments. They
are not passed to `forward()`.

## Shape Reference

For a single waveform with `batch=1`, `n_mels=128`, `n_mfcc=40`, and 163 time
frames:

```text
Elelet                -> (1, 128, 163) complex
EleSpectrogram        -> (1, 128, 163)
MaskedEleSpectrogram  -> (1, 128, 163)
EleCC                 -> (1, 40, 163)
MaskedEleCC           -> (1, 40, 163)
```

## Files

- `tf_transforms/transforms.py`: main transform classes.
- `tf_transforms/utils_harmonic_mask.py`: f0 CSV loading, harmonic masks, and
  reproducing-kernel smoothing.
- `tf_transforms/utils_elelet.py`: Elelet FFT convolution routines.
- `tf_transforms/utils_elespectrogram.py`: auditory filterbank construction.
- `demos/demo_tf.py`: comparison of STFT, mel, EleSpectrogram, Elelet, MFCC, and
  EleCC.
- `demos/demo_masked_tf.py`: comparison of regular and f0-masked
  representations.

## Practical Notes

- Use `torch.log(x + 1e-10)` before plotting magnitude-like outputs.
- Keep `n_fft`, `hop_length`, `f_min`, `f_max`, and `n_mels` the same when
  comparing regular and masked versions.
- Use `pad_mode="reflect"` when you want boundary behavior similar to STFT
  centering. Use `pad_mode="circular"` only when wraparound convolution is
  intentional.
- The masked transforms are only as good as the f0 contour. Check whether your
  contour file stores f0 or F1 before applying the optional `* 0.5` conversion.
