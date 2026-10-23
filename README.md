# Gradient Turnpike for Truncated Backpropagation Through Time

Numerical experiments for the ECC manuscript **“Truncated backpropagation through time induces a two-sided gradient turnpike property.”** A 16-state tanh RNN is fitted to the Wiener–Hammerstein benchmark by squared output simulation error.

## Gradient conventions

The model uses $x_{t+1}=f(x_t,u_t;\theta)$ and $y_t=h(x_t,u_t;\theta)$. All interval endpoints are inclusive.

| Symbol | Meaning |
|---|---|
| $\mathcal U=[a,b]$ | Parameter occurrences included in an update; length $b-a+1$. |
| $\mathcal C=[r,s]=[a-T_{\mathrm P},b+T_{\mathrm F}]$ | Truncated forward/backward context. |
| $T_{\mathrm P},T_{\mathrm F}$ | Past and future context extensions. |
| $g_t,\hat g_t$ | Time-local parameter-gradient contributions using full and truncated context. |
| $G_{\mathcal U},\widehat G_{\mathcal U}$ | Means of those contributions over $\mathcal U$. |

Each $g_t=\nabla_\theta l_t(x_t;\theta)+B_t^\top p_{t+1}$ includes its parameter occurrence's influence on subsequent losses. Full context uses the complete training sequence with $x_0=0,\ p_T=0$. Truncated context starts independently at $\hat x_r=0$ and ends with $\hat p_{s+1}=0$.

The gradient surrogates activate parameter occurrences only in $\mathcal U$. Outside it, parameter values are frozen while future state derivatives remain connected. Future losses in $\mathcal C$ therefore contribute even outside $\mathcal U$. This differs from differentiating losses restricted to $\mathcal U$. Mini-batches average independently initialized interval gradients.

## Setup and data

Use **Python 3.12** and run commands from the repository root. Shell examples use Bash/POSIX syntax. Create an environment and install the package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, create the environment with `py -3.12 -m venv .venv`, activate it with `.venv\Scripts\Activate.ps1` in PowerShell, and adapt the Bash training loop below.

[pyproject.toml](pyproject.toml) declares NumPy, PyTorch, Matplotlib, `nonlinear-benchmarks==1.0.1`, and optional pytest. Training defaults to CUDA when available; `--device cuda` requires a GPU, and `--device cpu` selects CPU. Training uses `float32`; gradient and contraction analyses use `float64`. CUDA is optional for installation and CPU workflows. For a specific CPU/CUDA build, install PyTorch using its [official selector](https://pytorch.org/get-started/locally/) before installing this package.

Run the complete test suite with `python -m pytest -q`; the `dev` extra supplies pytest. For headless plotting, set `MPLBACKEND=Agg`.

Archived results were verified with PyTorch `2.14.0+cu130`, Matplotlib `3.11.1`, and `nonlinear-benchmarks` `1.0.1`. A fresh Python `3.12.14` installation was checked with NumPy `2.5.4`, PyTorch `2.14.1` (CUDA `13.0`), Matplotlib `3.11.2`, and pytest `9.1.1`. These are verification versions, not additional pins.

The data come from the [SYSID 2009 Wiener–Hammerstein benchmark](https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein) by J. Schoukens, J. Suykens, and L. Ljung. Raw data are excluded because an explicit redistribution license was not located on the official pages. The loader calls `nonlinear_benchmarks.WienerHammerBenchMark(atleast_2d=True)`, which downloads and caches the original MAT file on first use:

```bash
python -c 'from gradient_turnpike.data import load_wh; d = load_wh(); print(len(d.train.u), len(d.validation.u), len(d.test.u))'
```

The expected lengths are `80000 20000 78800`. If the upstream download fails, obtain [WienerHammerBenchMark.mat from the official KTH HTTPS source](https://people.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat) and place it in the package cache: `~/.nonlinear_benchmarks/WienerHammerBenchMark/` on Linux.
On other platforms, `python -c 'from nonlinear_benchmarks.utilities import get_tmp_benchmark_directory; print(get_tmp_benchmark_directory())'` prints the cache root; put the MAT file in its `WienerHammerBenchMark/` subdirectory.
Subsequent runs can use the cache offline. Unnormalized CPU `float32` signals are split chronologically into 80,000 training and 20,000 validation samples. The separate official test sequence has 78,800 samples; test RMSE excludes its first 50 initialization samples. Training RMSE uses all training samples.

## Repository layout

| Path | Purpose |
|---|---|
| `src/gradient_turnpike/` | Model, simulation, gradients, training, evaluation, and contraction search. |
| `experiments/` | Experiment and figure command-line scripts. |
| `configs/wh_paper.json` | Seeds, context grid, and the separate training protocols. |
| `tests/` | Offline numerical and saved-data checks. |
| `outputs/analysis/` | Compact saved analysis tables and metadata. |
| `outputs/bptt/`, `outputs/tbptt/` | Five fixed-reference checkpoints and 25 Table I selected checkpoints. |
| `figures/` | Figure 2 (`gradient_turnpike`) and Figure 3 (`gradient_error_heatmap`), as PNG/PDF. |

`gradients.py` implements the occurrence-restricted surrogates and explicit pointwise adjoints. `simulation.py` reads out the current state. PyTorch's two trainable recurrent biases sum to the manuscript's $b$; gradient norms use both stored parameter blocks.

## Figures 2–3

**Render saved tables**, without checkpoints or gradient recomputation:

```bash
python experiments/make_paper_figures.py
```

Use `--figure gradient` for Figure 2 alone or `--figure gradient-error` for Figure 3. Outputs are `figures/gradient_turnpike.{png,pdf}` and `figures/gradient_error_heatmap.{png,pdf}`.

**Recompute tables from reference checkpoints** (replaces saved analysis data):

```bash
python experiments/analyze_gradient_turnpike.py --overwrite
python experiments/analyze_gradient_error_heatmap.py --overwrite
python experiments/make_paper_figures.py
```

These commands use the five included `final_model.pt` files under `outputs/bptt/run_seed_001/` through `run_seed_005/`, plus the benchmark data. No training is needed. Figure 2 uses seed 1, $\mathcal U=[40000,40299]$, and symmetric extensions $\{0,5,10,20,40\}$; its full reference uses all 80,000 training samples. Figure 3 uses five fixed models and the same 20 length-50 evaluation intervals for every context pair. Each cell is the median of 100 values of $\|\widehat G_{\mathcal U}-G_{\mathcal U}\|_2$. Tables are under `outputs/analysis/gradient_turnpike/pointwise/` and `outputs/analysis/gradient_error_heatmap/`.

**Repeat reference training from scratch** (optional; replaces included runs):

```bash
for seed in 1 2 3 4 5; do
  python experiments/train_bptt.py --seed "$seed" --run-id "seed_00$seed" --device cuda --overwrite
done
```

This protocol uses length-50 updates, 5,000 Adam steps, and validation every 50 steps. Learning rates are $10^{-3}$ through update 1500, $3\times10^{-4}$ through 3000, then $10^{-4}$. Figures use `final_model.pt` from update 5000; `best_model.pt` is validation-selected. Existing run directories require a new ID or explicit `--overwrite`.

Individual TBPTT runs use `experiments/train_tbptt.py --T-P 20 --T-F 20`. Both trainers support `--stop-after` and `--resume`. Resume checkpoints retain the current model/Adam/RNG state, best validation model, and accumulated history; recorded wall time excludes pauses between invocations.

## Numerical contraction assessment

The frozen five-model results are in `outputs/analysis/uniform_state_contraction/contraction_summary.csv` and `summary.json` (largest value found: 0.970756, rounded to 0.9708). To repeat the search using the included reference checkpoints and benchmark:

```bash
python experiments/analyze_uniform_state_contraction.py --paper
python experiments/analyze_uniform_state_contraction.py --smoke --seed 1
```

The CPU `float64` search evaluates 20-step Jacobian-product spectral norms on the fixed development input and initial states in $[-1,1]^{16}$. It screens all input starts and refines selected candidates by deterministic projected Adam ascent. This is numerical evidence, not a global certificate. In `outputs/analysis/uniform_state_contraction/`, reruns write separate `recomputed_contraction_summary.csv` or `smoke_contraction_summary.csv` files, leaving the frozen summaries intact.

## Separate training comparison: Table I

**Reconstruct the table from saved per-seed results**, without data or training:

```bash
python experiments/reproduce_training_table.py
```

**Rescore the 25 included validation-selected checkpoints** against the benchmark:

```bash
python experiments/reproduce_training_table.py --evaluate --device cuda
```

Use `--device cpu` if CUDA is unavailable. Rescoring checks training, validation, and test metrics; archived training times cannot be recovered from checkpoints.

**Repeat the full training comparison** (optional; replaces included results):

```bash
python experiments/run_training_comparison.py --paper --device cuda --overwrite
```

This retrains all 25 models and replaces the compact comparison outputs. BPTT uses full context with one length-1000 update interval; TBPTT uses length-50 intervals, batch size four, and symmetric extensions 0, 10, 20, and 40. Every run takes 5,000 staged-Adam steps and selects the lowest free-running validation MSE observed every 500 steps. Validation continues from replayed training inputs; test data are scored only after selection.

New checkpoints go under `outputs/{bptt,tbptt}/training_comparison_final/`; per-seed metrics summaries go to `outputs/analysis/training_comparison/`. Ratios are paired by seed before aggregation. Timings measure synchronized optimizer-update computation, excluding validation, serialization, and final scoring. Archived BPTT seeds 1–2 use deterministic timing repeats, identified in the saved rows.
