# Stated or Revealed Risk? IEEE Paper Source

This package contains the IEEE two-column manuscript, bibliography, vector figures, and figure-generation scripts.

## Main files

- `main.tex`: manuscript source
- `main.bbl`: compiled IEEE bibliography
- `references.bib`: BibTeX database
- `figures/*.pdf`: vector figures used in the manuscript
- `matlab/make_publication_figures.m`: MATLAB script reproducing the five publication figures from the verified result values
- `make_figures.py`: deterministic companion renderer used to generate the included vector PDFs in the current environment

## Build

A standard TeX Live installation with `IEEEtran`, `booktabs`, `tabularx`, `microtype`, `xurl`, and `cite` is sufficient.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript compiles to six pages. The length is intentional: the final editorial pass removes repeated result summaries, merges the reproducibility note into the empirical-design section, and retains only material needed to understand, assess, and reproduce the study.

## Figure note

Figure 4 is a forest plot of event-block bootstrap differences in precision-recall area. Points are mean incremental changes and horizontal lines are 95% bootstrap intervals; the dashed vertical line marks no change.

The included MATLAB script uses ordinary-language labels, MATLAB default publication colors together with line and marker distinctions, Times New Roman text, white backgrounds, and vector PDF export. MATLAB was not available in the execution environment used to assemble this package, so the checked-in vector PDFs were produced by the deterministic companion renderer using the same values, MATLAB color palette, and layout logic. Running the MATLAB script in MATLAB R2020a or later regenerates the corresponding figure files.
