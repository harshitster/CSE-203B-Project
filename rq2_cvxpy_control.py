"""
CVXPY Control Experiment: Solver-Neutral Primal vs Dual Comparison

Addresses the TA's concern about implementation differences:
the sklearn experiments compare liblinear (primal) vs libsvm (dual),
which are different codebases. This experiment uses CVXPY to solve
BOTH formulations with the same generic QP solver (SCS or ECOS),
so the timing difference reflects only formulation complexity.

We run on smaller datasets (CVXPY is much slower than specialized solvers)
and check whether the crossover direction matches the sklearn results.

Formulations used:
  Primal (kernelized via representer theorem):
    minimize    0.5 * beta^T K beta + C * sum(xi)
    subject to  y_i * ((K beta)_i + b) >= 1 - xi_i,  xi >= 0

  Dual:
    maximize    sum(alpha) - 0.5 * (alpha * y)^T K (alpha * y)
    subject to  0 <= alpha_i <= C,  sum(alpha_i * y_i) = 0

Both formulations use the same precomputed kernel matrix K,
so the only difference is the optimization problem structure.
"""

import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
import os

from utils import generate_synthetic_data


# =========================================================
# CVXPY SVM solvers
# =========================================================


def solve_primal_cvxpy(K, y, C, solver="SCS", verbose=False):
    """
    Solve the kernelized primal SVM using CVXPY.

    Primal (via representer theorem):
        minimize    0.5 * beta^T K beta + C * sum(xi)
        subject to  y_i * ((K beta)_i + b) >= 1 - xi,  xi >= 0

    Parameters:
        K: (n, n) kernel matrix
        y: (n,) labels in {-1, +1}
        C: regularization parameter
        solver: CVXPY solver name
    Returns:
        dict with solve_time, objective, beta, b, xi, status
    """
    n = K.shape[0]
    beta = cp.Variable(n)
    b = cp.Variable()
    xi = cp.Variable(n, nonneg=True)

    objective = cp.Minimize(0.5 * cp.quad_form(beta, cp.psd_wrap(K)) + C * cp.sum(xi))

    # y_i * ((K @ beta)_i + b) >= 1 - xi_i
    constraints = [cp.multiply(y, K @ beta + b) >= 1 - xi]

    prob = cp.Problem(objective, constraints)

    start = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose)
    solve_time = time.perf_counter() - start

    return {
        "solve_time": solve_time,
        "objective": prob.value,
        "beta": beta.value,
        "b": b.value,
        "xi": xi.value,
        "status": prob.status,
    }


def solve_dual_cvxpy(K, y, C, solver="SCS", verbose=False):
    """
    Solve the dual SVM using CVXPY.

    Dual:
        maximize    sum(alpha) - 0.5 * (alpha * y)^T K (alpha * y)
        subject to  0 <= alpha_i <= C,  sum(alpha_i * y_i) = 0

    Parameters:
        K: (n, n) kernel matrix
        y: (n,) labels in {-1, +1}
        C: regularization parameter
        solver: CVXPY solver name
    Returns:
        dict with solve_time, objective, alpha, status
    """
    n = K.shape[0]
    alpha = cp.Variable(n)

    # Q_ij = y_i * y_j * K_ij
    Q = np.outer(y, y) * K

    objective = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)))

    constraints = [
        alpha >= 0,
        alpha <= C,
        y @ alpha == 0,
    ]

    prob = cp.Problem(objective, constraints)

    start = time.perf_counter()
    prob.solve(solver=solver, verbose=verbose)
    solve_time = time.perf_counter() - start

    return {
        "solve_time": solve_time,
        "objective": prob.value,
        "alpha": alpha.value,
        "status": prob.status,
    }


# =========================================================
# Kernel computation
# =========================================================


def compute_kernel(X, kernel_type):
    """Compute the kernel matrix."""
    if kernel_type == "linear":
        return linear_kernel(X)
    elif kernel_type == "rbf":
        gamma = 1.0 / (X.shape[1] * X.var())
        return rbf_kernel(X, gamma=gamma)
    elif kernel_type == "poly":
        gamma = 1.0 / (X.shape[1] * X.var())
        return polynomial_kernel(X, degree=3, coef0=1, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")


# =========================================================
# Main experiment
# =========================================================


def run_control_experiment(
    n_samples_list,
    n_features_list,
    kernel_types,
    C=1.0,
    solver="SCS",
    n_runs=3,
    random_state=42,
):
    """
    Run the CVXPY control experiment.

    For each (n, d, kernel) setting, solve both primal and dual with
    the same CVXPY solver and record timing.
    """
    results = []

    total = len(n_samples_list) * len(n_features_list) * len(kernel_types)
    count = 0

    for n in n_samples_list:
        for d in n_features_list:
            nd_ratio = n / d
            for kernel_type in kernel_types:
                count += 1
                print(
                    f"[{count}/{total}] n={n}, d={d}, n/d={nd_ratio:.1f}, "
                    f"kernel={kernel_type} ... ",
                    end="",
                    flush=True,
                )

                X_train, _, y_train, _ = generate_synthetic_data(
                    n, d, noise=0.05, random_state=random_state
                )

                # Precompute kernel matrix (shared by both formulations)
                K = compute_kernel(X_train, kernel_type)

                # Ensure K is positive semidefinite (add small ridge)
                K = K + 1e-6 * np.eye(K.shape[0])

                try:
                    # Solve primal multiple times
                    primal_times = []
                    primal_obj = None
                    primal_status = None
                    for _ in range(n_runs):
                        res = solve_primal_cvxpy(
                            K, y_train.astype(float), C, solver=solver
                        )
                        primal_times.append(res["solve_time"])
                        primal_obj = res["objective"]
                        primal_status = res["status"]

                    # Solve dual multiple times
                    dual_times = []
                    dual_obj = None
                    dual_status = None
                    for _ in range(n_runs):
                        res = solve_dual_cvxpy(
                            K, y_train.astype(float), C, solver=solver
                        )
                        dual_times.append(res["solve_time"])
                        dual_obj = res["objective"]
                        dual_status = res["status"]

                    primal_time = np.median(primal_times)
                    dual_time = np.median(dual_times)
                    faster = "Primal" if primal_time < dual_time else "Dual"

                    # Duality gap: at optimum, primal_obj = dual_obj
                    # (primal is a minimization, dual is a maximization,
                    # and CVXPY returns the optimal value of each)
                    gap = (
                        abs(primal_obj - dual_obj) if primal_obj and dual_obj else None
                    )

                    print(
                        f"primal={primal_time:.4f}s [{primal_status}], "
                        f"dual={dual_time:.4f}s [{dual_status}], "
                        f"faster={faster}"
                    )

                    results.append(
                        {
                            "n_samples": n,
                            "n_features": d,
                            "nd_ratio": nd_ratio,
                            "kernel": kernel_type,
                            "primal_time": primal_time,
                            "dual_time": dual_time,
                            "primal_obj": primal_obj,
                            "dual_obj": dual_obj,
                            "duality_gap": gap,
                            "primal_status": primal_status,
                            "dual_status": dual_status,
                            "faster": faster,
                        }
                    )

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

    return results


# =========================================================
# Plotting
# =========================================================


def plot_control_scaling(results, output_dir="plots"):
    """
    Plot CVXPY primal vs dual timing, comparable to the sklearn plots.
    One subplot per kernel, x-axis = n/d ratio.
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = sorted(
            [r for r in results if r["kernel"] == kernel], key=lambda r: r["nd_ratio"]
        )

        ratios = [r["nd_ratio"] for r in subset]
        primal_times = [r["primal_time"] for r in subset]
        dual_times = [r["dual_time"] for r in subset]

        ax.plot(ratios, primal_times, "b-o", label="Primal (CVXPY)", markersize=6)
        ax.plot(ratios, dual_times, "r-s", label="Dual (CVXPY)", markersize=6)

        ax.set_xlabel("n/d Ratio")
        ax.set_ylabel("Solve Time (seconds)")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("CVXPY Control: Same Solver, Both Formulations", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cvxpy_control_scaling.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_control_duality_gap(results, output_dir="plots"):
    """
    Plot the duality gap from CVXPY solutions (also feeds into RQ1).
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = [
            r for r in results if r["kernel"] == kernel and r["duality_gap"] is not None
        ]
        if not subset:
            continue

        ratios = [r["nd_ratio"] for r in subset]
        gaps = [r["duality_gap"] for r in subset]

        ax.scatter(ratios, gaps, c="purple", edgecolors="k", s=60)
        ax.set_xlabel("n/d Ratio")
        ax.set_ylabel("Duality Gap |p* - d*|")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.suptitle("CVXPY Control: Duality Gap Verification", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cvxpy_control_gap.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_sklearn_vs_cvxpy_direction(cvxpy_results, sklearn_results, output_dir="plots"):
    """
    Compare the 'which is faster' direction between sklearn and CVXPY.
    Plots a grid showing agreement/disagreement.
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in cvxpy_results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]

        cvxpy_sub = sorted(
            [r for r in cvxpy_results if r["kernel"] == kernel],
            key=lambda r: r["nd_ratio"],
        )

        ratios = []
        cvxpy_ratios_list = []
        sklearn_ratios_list = []

        for cr in cvxpy_sub:
            # Find matching sklearn result
            sk_match = [
                r
                for r in sklearn_results
                if r["kernel"] == kernel
                and r["n_samples"] == cr["n_samples"]
                and r["n_features"] == cr["n_features"]
            ]
            if sk_match:
                sr = sk_match[0]
                ratios.append(cr["nd_ratio"])
                # Ratio > 1 means dual is slower (primal wins)
                cvxpy_ratios_list.append(
                    cr["dual_time"] / max(cr["primal_time"], 1e-10)
                )
                sklearn_ratios_list.append(
                    sr["dual_time"] / max(sr["primal_time"], 1e-10)
                )

        if not ratios:
            ax.text(0.5, 0.5, "No matching data", ha="center", transform=ax.transAxes)
            continue

        ax.plot(ratios, cvxpy_ratios_list, "g-^", label="CVXPY", markersize=7)
        ax.plot(ratios, sklearn_ratios_list, "m-v", label="sklearn", markersize=7)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Parity")

        ax.set_xlabel("n/d Ratio")
        ax.set_ylabel("Dual Time / Primal Time")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate: above line = primal faster, below = dual faster
        ax.text(
            0.02,
            0.98,
            "↑ Primal faster",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="blue",
            alpha=0.7,
        )
        ax.text(
            0.02,
            0.02,
            "↓ Dual faster",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            color="red",
            alpha=0.7,
        )

    plt.suptitle(
        "Direction Agreement: sklearn vs CVXPY\n(>1 = primal faster, <1 = dual faster)",
        fontsize=13,
    )
    plt.tight_layout()
    filepath = os.path.join(output_dir, "sklearn_vs_cvxpy_direction.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def print_control_table(results):
    """Print a summary table."""
    print(f"\n{'=' * 110}")
    print(
        f"{'Kernel':<10} {'n':>6} {'d':>6} {'n/d':>8} "
        f"{'Primal(s)':>10} {'Dual(s)':>10} {'Faster':>8} "
        f"{'Gap':>12} {'P_status':>12} {'D_status':>12}"
    )
    print("-" * 110)
    for r in results:
        gap_str = f"{r['duality_gap']:.2e}" if r["duality_gap"] is not None else "N/A"
        print(
            f"{r['kernel']:<10} {r['n_samples']:>6} {r['n_features']:>6} "
            f"{r['nd_ratio']:>8.1f} "
            f"{r['primal_time']:>10.4f} {r['dual_time']:>10.4f} {r['faster']:>8} "
            f"{gap_str:>12} {r['primal_status']:>12} {r['dual_status']:>12}"
        )


def print_direction_comparison(cvxpy_results, sklearn_results):
    """
    Print a side-by-side comparison of which solver is faster
    in CVXPY vs sklearn for matching settings.
    """
    print(f"\n{'=' * 90}")
    print("DIRECTION COMPARISON: sklearn vs CVXPY")
    print(f"{'=' * 90}")
    print(
        f"{'Kernel':<10} {'n':>6} {'d':>6} {'n/d':>8} "
        f"{'sklearn':>10} {'CVXPY':>10} {'Agree?':>8}"
    )
    print("-" * 90)

    agree_count = 0
    total_count = 0

    for cr in cvxpy_results:
        sk_match = [
            r
            for r in sklearn_results
            if r["kernel"] == cr["kernel"]
            and r["n_samples"] == cr["n_samples"]
            and r["n_features"] == cr["n_features"]
        ]
        if not sk_match:
            continue

        sr = sk_match[0]
        sk_faster = "Primal" if sr["primal_time"] < sr["dual_time"] else "Dual"
        cv_faster = cr["faster"]
        agree = sk_faster == cv_faster
        agree_count += agree
        total_count += 1

        print(
            f"{cr['kernel']:<10} {cr['n_samples']:>6} {cr['n_features']:>6} "
            f"{cr['nd_ratio']:>8.1f} "
            f"{sk_faster:>10} {cv_faster:>10} {'YES' if agree else 'NO':>8}"
        )

    print("-" * 90)
    if total_count > 0:
        print(
            f"Agreement: {agree_count}/{total_count} "
            f"({100 * agree_count / total_count:.0f}%)"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("CVXPY CONTROL EXPERIMENT")
    print("=" * 60)

    # Smaller datasets since CVXPY is much slower than sklearn
    n_samples_list = [50, 100, 200, 500]
    n_features_list = [10, 50, 200, 500]
    kernel_types = ["linear", "rbf", "poly"]
    C = 1.0

    results = run_control_experiment(
        n_samples_list=n_samples_list,
        n_features_list=n_features_list,
        kernel_types=kernel_types,
        C=C,
        solver="SCS",
        n_runs=3,
    )

    print_control_table(results)

    # Generate plots
    plot_control_scaling(results)
    plot_control_duality_gap(results)

    print("\nCVXPY control experiment done! Plots saved to ./plots/")
