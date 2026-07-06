# Automatic rumble F0 for DDSP

## Recommended algorithm

The automatic tracker is deliberately a composition of existing ideas rather
than a new estimator:

1. Compute a 16 ms-hop Elelet representation from 5--500 Hz with 1024 channels.
2. Track one visible peak path in each of four restricted H1 bands: 17--30,
   22.5--40, 30--58, and 50--82 Hz.
3. Usually retain the globally dominant H1 ridge. If that ridge is below
   22.5 Hz, use an Elelet-space SHRP score to decide whether it is a genuinely
   low H1 or the physical fundamental of a higher call. This confines SHRP to
   the octave decision; it cannot jump to an arbitrary upper partial.
4. Search only +/-0.5 Hz around the selected coarse path. Refine on summed
   Elelet power at harmonics 2--24 of `F0 = H1 / 2`, weighted by
   `1 / harmonic**0.35`. Higher partial displacement therefore supplies
   sub-bin FM information even though H1 itself has coarser channel spacing.
5. Find the final path with a bounded, lightly quadratic Viterbi transition
   penalty and quadratic sub-grid interpolation. No median filter is applied.

The CSV output contains only `time` and `frequency`. By default, `frequency`
keeps the project-compatible H1 convention; DDSP F0 is `frequency / 2`.

## Tuned Elelet representation

At 16 kHz, the selected settings are:

```text
kernel_size = 16000
supp_mult = 0.3
hop_length = 256
num_channels = 1024
f_min = 5 Hz
f_max = 500 Hz
scale = elelog
```

The effective windows are approximately 0.650 s at 20 Hz, 0.475 s at 40 Hz,
0.370 s at 100 Hz, and 0.335 s at 200 Hz. The 24k/0.2 setting is a sensible
low-frequency analysis bank, but it was not the best DDSP tracking bank: the
shorter maximum kernel preserved time-varying modulation better.

## Evaluation

The representation grid used 48 calls stratified over the complete corrected
H1 range. The selected setting was then frozen and evaluated on all 1,469 files
in `rumbles/train`, against `f0_corrected`.

| method | F0 MAE (Hz) | median abs cents | within 25 cents | gross >200 cents | FM correlation | H1 step MAE (Hz) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid Elelet | 0.881 | 73.25 | 0.749 | 0.098 | 0.560 | 0.071 |
| fixed-band peak (22.5--50 Hz) | 1.004 | 92.88 | 0.764 | 0.115 | 0.571 | 0.100 |
| broad-band peak (17--82 Hz) | 1.547 | 169.98 | 0.732 | 0.164 | 0.568 | 0.099 |

The hybrid is the safer DDSP default: relative to the existing fixed-band peak
tracker it reduces mean F0 error by 12%, gross-frame errors by 15%, median
absolute cents by 21%, and frame-step error by 29%. The trade-off is 1.5
percentage points less strict +/-25-cent accuracy and slightly lower mean FM
correlation. Per file, however, the hybrid has better FM correlation on 830 of
1,469 calls and lower step error on 1,340 calls; its lower mean FM correlation
is driven by the remaining octave-selection failures.

High-H1 calls above 50 Hz and overlaps remain the main failure modes. The
tracker computes local and proposal confidence internally, but the deliberately
minimal contour CSV does not store them.

Reproduce the grid and evaluation with:

```bash
python analysis/evaluate_hybrid_elelet.py --mode grid --sample-size 48 \
  --kernel-sizes 16000,24000,32000 --supp-mults 0,0.1,0.2,0.3,0.4

python analysis/evaluate_hybrid_elelet.py --mode evaluate \
  --kernel-sizes 16000 --supp-mults 0.3
```
