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

Future extractors should follow the same pattern, for example:

```text
extract_f0_pyin.py  -> f0_pyin/
extract_f0_swift.py -> f0_swift/
```

Precompute is shared because the annotator can switch between STFT and Elelet
views:

```bash
python f0_extraction/precompute_representations.py --input data/rumbles
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

Refinement is currently a single development script that reads `f0_corrected`
independently of the extraction algorithm and writes:

```bash
python f0_extraction/refine_f0.py --input data/rumbles
```

```text
f0_refined/
```

`frequency` currently stores the tracked first harmonic (F1/H1) by default.
Refinement converts H1 to F0 internally, refines in F0 space, and writes H1
back out.
