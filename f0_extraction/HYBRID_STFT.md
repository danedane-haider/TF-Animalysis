# Multi-resolution STFT rumble tracker

This is the STFT counterpart of the hybrid Elelet tracker. It keeps the same
restricted-band proposal, SHRP octave selection, harmonic evidence, and smooth
Viterbi structure, while allowing the proposal and evidence stages to use
different physical windows.

For audio `x[n]`, compute aligned STFT magnitudes

```text
X_c(k,t) = |STFT(x; L_c, N_c, H)|
X_f(k,t) = |STFT(x; L_f, N_f, H)|,
```

where `L` is the physical Hann-window length, `N` is the FFT size, and the
common hop is `H=256` samples (16 ms at 16 kHz). Zero padding to `N=32768`
densifies the frequency grid but does not increase physical resolution.

For each restricted H1 band `B_j`, the coarse ridge proposal is

```text
p_j(t) = argmax_{f in B_j, |f-p_j(t-1)| <= 1.5 Hz} X_c(f,t).
```

The proposal is ranked with fine-window SHRP and anchor evidence. Given an H1
candidate `h`, harmonic evidence is accumulated at `m h / 2`:

```text
E(h,t) = sum_m m^-0.35 R_f(mh/2,t) - 0.30 I_f(h,t),
```

where `R_f` is local harmonic ridge power and `I_f` penalizes energy between
the expected harmonics. The winning proposal restricts candidates to
`|h-p(t)| <= 1 Hz`. The final path minimizes

```text
J(h_1:T) = -sum_t z(E(h_t,t))
           + 0.04 sum_t ((h_t-h_{t-1}) / 0.1 Hz)^2,
```

with jumps above 1.5 Hz forbidden. The DDSP control is `f0(t)=h(t)/2`.

## Usage

```bash
python f0_extraction/extract_f0.py \
  --input /path/to/rumbles \
  --method hybrid_stft
```

The evaluated robust defaults are `L_c=L_f=8192`, `N_c=N_f=32768`, harmonic
bin radius 2, and refinement radius 1 Hz. The implementation remains
multi-resolution: a long-proposal preset is `L_c=12000`, `L_f=8192`:

```bash
python f0_extraction/extract_f0.py \
  --input /path/to/rumbles \
  --method hybrid_stft \
  --hybrid_stft_coarse_win_length 12000 \
  --hybrid_stft_fine_win_length 8192
```

## Training-set evaluation

Against 1469 `f0_corrected` contours (356026 frames), using macro-averaged
per-file metrics:

| Setting | F0 MAE (Hz) | Median error (cents) | Within 25 cents | Gross >200 cents | FM correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| STFT robust, 8192/8192 | 0.8650 | 72.44 | 75.98% | 9.48% | 0.5989 |
| STFT long-proposal, 12000/8192 | 0.9826 | 84.13 | 74.93% | 10.90% | 0.5914 |
| Hybrid Elelet, 16000/0.3 | 0.8811 | 73.25 | 74.87% | 9.78% | 0.5600 |

The robust STFT setting is slightly better overall on this reference set. The
true multi-resolution setting remains useful as an explicit experimental
preset, but its tuning-subset FM advantage did not fully carry over to all
files. Because the corrected contours originated in an Elelet/manual workflow,
also judge harmonic alignment and DDSP resynthesis quality, not only reference
error.
