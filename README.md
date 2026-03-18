# An Empirical Study of SVMs: Solver Scaling Across Kernels and Numerical Precision of Primal-Dual Equivalence

**Course:** CSE-203B Machine Learning  
**Authors:** Charvi Bannur, Harshit Timmanagoudar, Saniya Patil, Rohan Thorat

## Overview

This project empirically investigates two practical gaps in SVM theory:

**Research Question 1 — Solver Scaling Crossover:** Where is the empirical crossover point at which the primal formulation becomes faster than the dual (and vice versa), and how does kernel choice shift it?

**Research Question 2 — Numerical Precision:** How large is the primal-dual gap in practice, and what factors (regularization, noise, dimensionality, kernel) cause it to grow?

## Key Findings

- **Crossover ordering:** The primal becomes faster than the dual at n/d ≈ 1 (linear), n/d ≈ 3.6 (RBF), and n/d ≈ 5.5 (polynomial), confirming that kernel complexity shifts the crossover in favor of the dual.
- **Formulation vs implementation:** A CVXPY control experiment shows the dual is structurally faster in all 48 settings when implementation is held constant. The primal's speed advantage in sklearn is driven by liblinear's optimized coordinate descent, not the formulation itself.
- **Two-regime precision:** The primal-dual gap is small (10⁻⁹ to 10⁻²) when the solver converges, but becomes meaningless under non-convergence. The regularization parameter C is the dominant factor.
- **Kernel robustness:** RBF is the most numerically stable kernel; linear is the most fragile at high C.

## Project Structure

```
.
├── run_all.py                  # Main runner for all experiments
├── solver_scaling.py           # RQ1: Primal vs dual timing experiments
├── cvxpy_control.py            # RQ1: CVXPY control experiment (same solver, both formulations)
├── numerical_precision.py      # RQ2: Primal-dual gap and KKT analysis
├── visualize_boundaries.py     # Decision boundary visualization
├── utils.py                    # Shared utilities (data generation, timing, objectives, KKT)
├── plots/                      # Generated figures (created on first run)
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/harshitster/CSE-203B-Project.git
cd CSE-203B-Project
pip install -r requirements.txt
```

## Usage

### Run all experiments

```bash
python run_all.py
```

### Run individual experiments

```bash
python run_all.py rq1       # Solver scaling crossover (RQ1 in paper)
python run_all.py rq2       # Numerical precision (RQ2 in paper)
python run_all.py cvxpy     # CVXPY control experiment
python run_all.py rq1full   # Solver scaling + CVXPY with direction comparison
python run_all.py viz       # Decision boundary visualization
```

### Run files directly

```bash
python numerical_precision.py
python solver_scaling.py
python cvxpy_control.py
python visualize_boundaries.py
```

All plots are saved to the `plots/` directory.

## Experiments

### Solver Scaling (RQ1)

- **Experiment A:** Fix d=100, vary n ∈ {50, 100, 200, 500, 1000, 2000, 5000, 10000}
- **Experiment B:** Fix n=1000, vary d ∈ {10, 25, 50, 100, 200, 500, 1000, 2000, 5000}
- **Kernels:** Linear, RBF, Polynomial (degree 3)
- **Primal solver:** LinearSVC(dual=False) with Nyström approximation for nonlinear kernels
- **Dual solver:** LinearSVC(dual=True) for linear; SVC for nonlinear kernels
- **Fairness:** Adaptive accuracy matching ensures the Nyström approximation matches dual accuracy within 3% before timing comparison
- **CVXPY control:** Both formulations solved with the same SCS solver to isolate formulation effects

### Numerical Precision (RQ2)

- **Full factorial design:** 4 sample sizes × 3 feature dimensions × 3 noise levels × 4 C values × 3 kernels = 432 settings
- **Metrics:** Absolute gap, relative gap, KKT complementary slackness violations, dual feasibility violations
- **Convergence tracking:** Solver convergence status captured to distinguish precision limitations from solver failures

## Dependencies

- numpy
- scikit-learn
- matplotlib
- cvxpy

## References

1. Boyd, S. and Vandenberghe, L. *Convex Optimization*. Cambridge University Press, 2004.
2. Cortes, C. and Vapnik, V. "Support-Vector Networks." Machine Learning, 1995.
3. Schölkopf, B. and Smola, A.J. *Learning with Kernels*. MIT Press, 2002.
4. Chang, C.-C. and Lin, C.-J. "LIBSVM: A Library for Support Vector Machines." ACM TIST, 2011.
5. Platt, J. "Sequential Minimal Optimization." Microsoft Research, 1998.
6. Williams, C.K.I. and Seeger, M. "Using the Nyström Method to Speed Up Kernel Machines." NeurIPS, 2001.
7. O'Donoghue, B. et al. "Conic Optimization via Operator Splitting." JOTA, 2016.