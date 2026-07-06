# F0/F1 Contour Workflow

Extraction is algorithm-specific. Each extractor writes its own `f0_<algorithm>`
directory so different methods can be compared later.

```bash
python f0_extraction/extract_f0_elelet.py --input data/rumbles
python f0_extraction/extract_f0_stft.py --input data/rumbles
```

The default outputs are:

```text
f0_elelet/
f0_stft/
```

For automatic DDSP-oriented extraction on noisy rumbles, use the hybrid
Elelet tracker:

```bash
python f0_extraction/extract_f0.py \
  --input data/rumbles \
  --method hybrid_elelet
```

It writes only `time` and `frequency`; `frequency` follows the existing H1
convention. Its tuned defaults at 16 kHz are `kernel_size=16000` and
`supp_mult=0.3`. See
[`HYBRID_ELELET.md`](HYBRID_ELELET.md) for the algorithm and evaluation.

The default output is `f0_hybrid_elelet/`. To write DDSP F0 rather than H1 in
the `frequency` column, add `--divide_by_2`. The dedicated
`extract_f0_hybrid_elelet.py` entry point remains available as a thin
algorithm-specific alternative.

Saving is enabled by default for `hybrid_elelet`. `extract_f0.py` first checks
for a nearby cache with matching transform metadata, then runs the existing
`precompute_representations.py` Elelet workflow with the hybrid parameters,
writes/fills its canonical cache, and tracks F0 from that cache. Use
`--no-save_elelet_representations` for an in-memory-only run.
Start correction directly in the cached view with:

```bash
python f0_extraction/annotate_f0.py \
  --input data/rumbles \
  --f0_dir f0_hybrid_elelet \
  --initial_spec_mode elelet
```

The corresponding multi-resolution STFT implementation is available through:

```bash
python f0_extraction/extract_f0.py \
  --input data/rumbles \
  --method hybrid_stft
```

The robust default uses 8192-sample proposal and evidence windows with a
32768-point FFT. For a genuinely multi-resolution, long-proposal setting, use
`--hybrid_stft_coarse_win_length 12000 --hybrid_stft_fine_win_length 8192`.
See [`HYBRID_STFT.md`](HYBRID_STFT.md) for the equations and evaluation.

Future extractors should follow the same pattern, for example:

```text
extract_f0_pyin.py  -> f0_pyin/
extract_f0_swift.py -> f0_swift/
```

By default, extractors compute representations on the fly. You can precompute
STFT and Elelet once, including a `metainfo.json` with the transform parameters,
then point any spectral extractor at the saved folder:

```bash
python f0_extraction/precompute_representations.py --input data/rumbles
python f0_extraction/extract_f0_elelet.py --input data/rumbles --use_precomputed_representations data/rumbles/elelet_750
python f0_extraction/extract_f0_stft.py --input data/rumbles --use_precomputed_representations data/rumbles/stft_750
```

Elelet amplitudes can be sampled along any existing contour directory after
precomputing representations:

```bash
python f0_extraction/extract_f0_magnitude_elelet.py --input data/rumbles --use_precomputed_representations data/rumbles/elelet_750
```

Annotation is also shared. Pick which extracted contour is editable with
`--f0_dir`; every other available `f0*` directory for the current file is
plotted as a comparison overlay, including refined outputs.

```bash
python f0_extraction/annotate_f0.py --input data/rumbles --f0_dir f0_elelet
python f0_extraction/annotate_f0.py --input data/rumbles --f0_dir f0_stft
```

Inside the annotator, press `F` to toggle all F0 contour overlays, use the
right-side contour checkboxes or number keys (`1`, `2`, `3`, ...) to toggle
individual contour versions, and press `H` to toggle harmonic overlays. Harmonic
overlays are computed for every enabled contour from `F0 = H1 / 2`, because
contour CSV frequencies store H1 in this workflow. The annotator plots both the
CSV H1 contour and its derived F0 line.

Manual corrections always converge into:

```text
f0_corrected/
```

Refinement reads `f0_corrected` by default and writes:

```bash
python f0_extraction/refine_f0_elelet.py --input data/rumbles
python f0_extraction/refine_f0_elelet.py --input data/rumbles --use_precomputed_representations data/rumbles/elelet_750
```

```text
f0_refined/
```

`frequency` currently stores the tracked first harmonic (F1/H1) by default.
Refinement converts H1 to F0 internally, refines in F0 space, and writes H1
back out.

## Resynthesize a photographed drawing

Print `output/pdf/f0_drawing_template_a4.pdf`. The German A4 landscape sheet
maps the bottom edge to 10 Hz and the top edge to 30 Hz on a linear frequency
scale. Photograph it upright in landscape orientation with the full blue
rectangle visible. Then provide only the photo and a reference rumble:

```bash
.venv/bin/python scripts/resynthesize_drawn_f0.py \
  /path/to/drawing.HEIC \
  /path/to/reference_rumble.wav
```

The script crops the blue frame, tracks the largest continuous dark line,
linearly maps it to 10-30 Hz, and interpolates it over the exact reference-audio
duration at the checkpoint frame rate. It discovers the reference rumble's
original contour from sibling `f0_*` directories, preferring
`f0_corrected_refined`; if none exists, it extracts one automatically.

Elephant-Synth first analyzes the reference audio with that **original F0** to
compute its amplitude envelope, harmonic distribution, learned phases, and
noise controls. Those controls are then reused unchanged while the harmonic
oscillator and final F0-dependent filtering are rerun with the **drawn F0**.
The filtered-noise branch is attenuated to `0.1` gain by default.
The drawn CSV writes Elephant-Synth's expected H1 column
(`frequency = 2 * f0_hz`) and invokes
`/Users/dani/Documents/GitHub/Elephant-Synth/resynth/resynth.py` with the
`baseline_phase_02/pth-best` checkpoint.

Outputs are written below `output/drawn_resynthesis/<photo>__<rumble>/`:

- `drawn_f0_resynthesis.wav`: final reconstructed audio
- `drawn_f0_resynthesis_spectrogram.png`: 8192-sample Hann-window spectrogram from 0-250 Hz
- `f0_original/*.f0.csv`: original F0 used for bottleneck analysis
- `f0_drawn/*.f0.csv`: interpolated F0/H1 control contour
- `diagnostics/03_extracted_contour_overlay.png`: red extraction overlay
- `manifest.json`: paths, parameters, and the exact resynth command

Use `--extract-only` to inspect the overlay and CSV without running the model.
