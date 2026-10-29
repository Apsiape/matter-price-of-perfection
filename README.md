# Matter that pays the price of perfection

Seth Douglas and Nidhal Mghirbi.

Version 1.0.0, October 2026. DOI: [10.5281/zenodo.23771444](https://doi.org/10.5281/zenodo.23771444). Concept DOI (all versions): [10.5281/zenodo.23771443](https://doi.org/10.5281/zenodo.23771443). A preprint; it has not been peer reviewed.

We construct translation-invariant nearest-neighbour lattice families whose fixed-coupling thermodynamic gap stays positive while the entanglement entropy of a growing cell-anchored region has any prescribed nonnegative limit-computable logarithmic coefficient. Ordered lower and upper coefficients have the exact second-level one-sided arithmetic classes within an effective cell model. Separately, a referee history cell converts low energy into a nonlocal-game strategy with identical pure-state Schmidt coefficients and a mixed-state distillable-entanglement certificate. The extraction cost is controlled by the clock grid's corner-to-uniform resistance: linear, logarithmic, and bounded for one, two, and three clocks. We give a conditional material theorem with explicit layout and selection hypotheses, a worst-case whole-cell perturbation law, and necessary temperature, preparation-time, and work bounds. Disjoint quasi-local spin-lattice measurements yield only the closure of finite-dimensional correlations. All gap limits use factorial tori before the coupling limit; the entanglement cut grows inside a selected cell. The game-driven material construction remains conditional.

The manuscript and source are in paper/. The self-contained submission source is
paper/arxiv/. The sequential finite checks, recorded outputs and checksums are
in paper/anc/. The complete game-driven material theorem has explicit layout
and positive-energy selection hypotheses; its local implementation is not claimed.

Build outside the source directory with:

    python paper/build.py --output-dir BUILD_DIRECTORY

Run the finite checks with Python, NumPy, SciPy, psutil and threadpoolctl installed:

    python paper/anc/run_checks.py
    python paper/anc/verify_hashes.py

Builds use pdflatex and BibTeX. The build script rejects final-pass warnings,
undefined references, and overfull or underfull boxes. The submission directory
can also be built with pdflatex main, bibtex main, and three further pdflatex passes.

## Related
- "Uncomputable critical exponents" (Seth Douglas and Nidhal Mghirbi), version 2.0.0, [10.5281/zenodo.23757361](https://doi.org/10.5281/zenodo.23757361).
- "The uncomputable price of precision" (Seth Douglas and Nidhal Mghirbi), version 4.0.0, [10.5281/zenodo.23753741](https://doi.org/10.5281/zenodo.23753741).
- "The price of precision" (Seth Douglas and Nidhal Mghirbi), version 1.1.0, [10.5281/zenodo.23753349](https://doi.org/10.5281/zenodo.23753349).

The text and documentation are CC BY 4.0. Scripts are MIT or Apache 2.0 at your
option. See RIGHTS.md.
