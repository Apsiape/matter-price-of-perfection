Finite consistency checks

Run python run_checks.py with NumPy, SciPy, psutil and threadpoolctl installed.
The runner executes children sequentially, sets all supported numerical-library
thread limits to one, and monitors combined parent/child memory below 900 MB.
Recorded outputs and summary.json describe the completed run.
Run python verify_hashes.py to check every shipped ancillary file.

check_fk_lemma.py tests 68 elementary two-clock ground-state instances.
check_ruler_and_pairs.py samples perfect-stripe minimizers and perturbed pairs.
check_fk_log.py tests resistance extraction on 160 low-energy states.
check_clock_dimension.py tests 48 three-clock low-energy states and finite sums.
independent_checks.py tests rational identities, small complex operator inequalities,
path and grid pseudoinverses, reference estimates, pair-field formulas, thermal
pair identities, depth selection, and numerical-library thread counts.

Floating-point eigensystems are consistency checks, not certified interval proofs.
No finite check certifies an uncomputable limit, local referee rule table,
positive-energy cell charging hypothesis, or generic local-noise model.
