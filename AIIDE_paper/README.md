# AIIDE 2026 submission

`main.tex` is a full paper targeted at the **AIIDE 2026 main technical track**,
written to follow Matthew Guzdial's "How to Write an AIIDE Paper" guide
(problem-first introduction with explicit contributions, a related-work section
organized by area that differentiates the work, a replicable methodology, an
argument for why the evaluation is appropriate, a walk-through of results, and a
discussion with limitations and future work).

**Title:** *A Gold-Standard Study of What Makes a Lightweight Game-Playing Agent
Strong*

## Build

```bash
export PATH="/apps/generic/texlive/2026/bin/x86_64-linux:$PATH"
pdflatex main.tex && pdflatex main.tex     # two passes for author-year citations
```

No `bibtex` step is needed: the bibliography is a hand-checked
`thebibliography` block so every entry is correct and correctly mapped. The
paper compiles to about 10 pages: **8 pages of content** (comfortably within
AIIDE's 9-page limit, leaving room for the official template to reflow) plus the
references, which do not count toward the limit. All six figures are full-width
vector PDFs so the charts and numbers stay legible.

## Before you submit (a short checklist)

1. **Format.** AIIDE 2026 requires the official **AAAI Press author kit** (AAAI
   two-column). The preamble here is a faithful, self-contained approximation so
   the paper builds anywhere. Drop in the official `aaai2026.sty` and
   `aaai2026.bst` from the kit and replace the marked preamble block; the body
   and the bibliography are written to transfer without changes.
2. **Do not post on CEUR.** The CFP states that AIIDE 2026 treats CEUR as an
   archival venue and will reject papers available there. (CEUR is only for one
   of the workshops, not the conference.) **arXiv is allowed.**
3. **Double-blind.** The author block is controlled by a one-line toggle near the
   top of `main.tex`. It currently **shows the real authors** (`\anonymousfalse`:
   Nima Kelidari, Mohammadsaeed Haghi, Mahdi Salmani, USC). For the AIIDE
   submission, change that single line to `\anonymoustrue` to print "Anonymous
   Submission" instead. Also double-check the figures contain no identifying text
   (the figure generator has none).
4. **Dates (from the official CFP).** Abstract due **June 19, 2026**; full paper
   due **June 26, 2026**; notification **August 7, 2026**; conference
   **November 9 to 13, 2026**, Belo Horizonte, Brazil.

## Figures

The seven figures in `figures/` are copied from `../paper/figures/` and are
generated from saved evaluation logs by `../paper/make_figures.py`. To refresh
them, regenerate there and re-copy.

## What is claimed, and what is not

The "gold standard" is described honestly as a strong, fixed, deterministic
**expert**: it solves meld decomposition exactly (the one provably optimal
piece) and otherwise plays principled endgame heuristics. It is **not** a
game-theoretic optimum for the full imperfect-information game, and the paper
says so. "Win-rate against the expert" is therefore a fixed, reproducible
yardstick, not a distance from perfect play. Every number in the paper traces to
measured logs under `../sweep/`.
