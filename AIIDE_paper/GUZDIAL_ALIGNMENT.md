# Guzdial alignment + green/red-flag checklist

Self-audit of `main.tex` against Matthew Guzdial's "How to Write an AIIDE Paper"
(https://www.guzdial.com/blog/how-to-write-an-aiide-paper/) and the accepted-/rejected-paper
intersection. Updated for the Phase-8 (architecture + baselines + second-game) rewrite.

## Structure (Guzdial section-by-section)

| Guzdial expectation | Where in paper | State |
|---|---|---|
| Intro starts at the specific field, not "games are important" | §1 opens on imperfect-information card games + the two concrete gaps (opponent bottleneck, no cheap yardstick) | ✅ |
| One clear problem/lack | §1: no cheap absolute yardstick → choices tuned against weak/moving references | ✅ |
| Bulleted contributions, each traceable to evidence | §1 list of 5 bullets; each maps to a results subsection | ✅ |
| Related work in differentiating subsections (≥20 refs) | §2: six paragraphs (deep RL for card games; self-play/leagues; search & equilibrium; architectures; reward/imitation; LLMs; masking/tooling), **36 references**, each para ends with how we differ | ✅ |
| Methodology replicable, every decision justified | §3 setup, §4 expert (exact vs heuristic split is explicit), §5 methods studied (incl. why TRPO mask is finite not −∞) | ✅ |
| Experiments argue *appropriateness* of the metric | §6: one metric (win% vs fixed expert) justified by strong/identical/cheap; rliable IQM+CIs for multi-seed | ✅ |
| Results summarized + walked through; surprises highlighted | §7 (expert never gins; can't pay into gin), §8 (fair search is weak; info-bound ceiling) | ✅ |
| Discussion/Limitations/Future Work is defensive, anticipates criticism | §9: information-bound ceiling, limitations (one game, heuristic expert, reactive policy), future work (opponent-hand inference) | ✅ |
| Conclusion = 2–4 takeaways | §10: recipe that works, info-bound ceiling, methodology lesson | ✅ |
| Visual pitch of the method | Fig. 1 system overview; Fig. 6 arch+ISMCTS | ✅ |

## Green flags applied (accepted-paper traits)

- **Targeted ablations isolating what matters** — one factor at a time, graded on one fixed number.
- **Statistical rigor** — 2000-game CIs for the headline; IQM + stratified bootstrap CIs over seeds for the architecture sweep (rliable; Henderson et al.).
- **Strong baselines** — determinized ISMCTS (fair) + oracle upper bound + NFSP + CFR, the #1 reject reason pre-empted.
- **≥1 negative result** — learned embeddings, DAgger, dense rewards, live LLM, *and now* "architecture doesn't help" and "fair search is weak / NFSP undertrained."
- **Generality** — second game (Leduc) with a computable optimum; game-agnostic released pipeline.
- **Honest scoping** — "expert," never "optimal"; oracle clearly labeled as an upper bound, not the headline.
- **Reproducibility** — every figure from saved logs; W&B curves; released code/configs/seeds.

## Red flags checked and avoided

- **Weak/missing baselines** → search + NFSP + CFR added.
- **Single-config overfitting** → Stage-C shows the architecture ranking holds across PPO/PFSP recipes.
- **No stats / point estimates** → IQM + bootstrap CIs; CIs reported as overlapping (no over-claim of a "winner" architecture).
- **Overclaiming** → the information-bound ceiling is stated as a measurement, backed by three independent lines of evidence; no "solves/perfect."
- **Failure to identify source of (non-)gains** → the 26%-fair vs 85%-oracle gap names the source as hidden information.
- **Single-domain** → Leduc generality + game-agnostic framing.

## Consistency guard (abstract ↔ body)

The submitted abstract is **unchanged**. It already promised: architecture comparison, NFSP + information-set Monte-Carlo baselines, Leduc generality, robust statistics, game-agnostic release. The Phase-8 rewrite makes the body deliver each of these, so abstract and body now match (previously the body lagged the abstract).

## Open items before camera-ready

- **Official AAAI style applied.** The paper now uses the real `aaai24.sty`/`.bst` from the AAAI
  author kit (`AuthorKit24-4.zip`), anonymous via `\usepackage[submission]{aaai24}`. Under the
  official style the content is **8 pages** (references follow, pages 8–10), comfortably within the
  9-page limit. No forbidden packages; compiles clean with no undefined refs or overfull boxes.
- **De-anonymize for camera-ready only:** remove the `[submission]` option and fill in
  `\author{}`/`\affiliations{}` with the real names/affiliation.
- Optional: the official bibliography style (`aaai24.bst`) is set; the reference list is currently a
  manual `thebibliography` (renders correctly via natbib). For the camera-ready, converting the 36
  entries to a `.bib` + `\bibliography{}` would let `aaai24.bst` format them exactly.
- Content has ~1 page of slack, so a dropped visual (e.g., the milestone "journey" plot) can be
  restored if desired.
