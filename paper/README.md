# Phase-1 sweep paper

4-page NeurIPS-preprint paper reporting the Phase-1 PPO hyperparameter
sweep against `RandomAgent`.

## Files

- `main.tex` — paper source. Imports `neurips_2024.sty` (same style as the
  midterm report).
- `refs.bib` — minimal bibliography.
- `make_figures.py` — regenerates every figure in `figures/` from
  `../sweep/results/*.json`. Run with the project env:
  `PYTHONNOUSERSITE=1 /scratch1/kelidari/envs/coev/bin/python paper/make_figures.py`.
- `figures/` — png + pdf for each figure, plus `sweep_summary.csv`.

## To compile

`pdflatex` isn't available on the login node here; compile elsewhere.

1. Drop the same `neurips_2024.sty` you used for the midterm into this
   directory (or place `main.tex` next to it).
2. Compile:
   ```bash
   pdflatex main.tex
   bibtex   main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. The expected output is exactly 4 pages including bibliography. If
   `placeins`/`subcaption` push a figure to a fifth page, the easiest fix
   is to tighten the column widths in the two `\begin{subfigure}` blocks
   inside `\section{Results}`.

## If you change the sweep

Regenerate the figures **before** rebuilding the PDF:
```bash
PYTHONNOUSERSITE=1 /scratch1/kelidari/envs/coev/bin/python paper/make_figures.py
```
The numbers cited inline in `main.tex` (98.3–99.6 % win rate, 0.510–0.541
mean reward, the per-config rows of Table 1) are pulled from the current
`sweep/results/*.json` snapshot; if you re-run the sweep, update those
values too.
