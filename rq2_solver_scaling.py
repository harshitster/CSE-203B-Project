"""
Research Question 2: Solver Scaling Crossover

Finds the empirical crossover point where primal solvers become faster
than dual solvers (or vice versa) as the sample-to-feature ratio (n/d)
varies, and examines how kernel choice shifts this crossover.

For nonlinear kernels (RBF, poly), uses an adaptive approach to find
the minimum n_components needed for the primal approximation to match
dual accuracy within a tolerance, ensuring a fair timing comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import warnings
import os

from utils import generate_synthetic_data, time_fit

warnings.filterwarnings("ignore")


# =========================================================
# Solver wrappers
# =========================================================


def get_primal_solver(kernel_type, C, n_components=300, random_state=42, gamma=None):
    """
    Return a primal solver pipeline for the given kernel.

    - Linear: LinearSVC(dual=False)
    - RBF: Nystroem(kernel='rbf') -> LinearSVC(dual=False)
    - Poly: Nystroem(kernel='poly') -> LinearSVC(dual=False)

    For RBF/poly, gamma should be precomputed from the training data
    to match SVC(gamma='scale'). Pass gamma explicitly.

    Note: Both RBF and poly use Nystroem, which gives better kernel
    approximation than RBFSampler (random Fourier features) but is
    capped at n_train components.
    """
    if kernel_type == "linear":
        return LinearSVC(
            C=C, dual=False, max_iter=50000, tol=1e-6, random_state=random_state
        )
    elif kernel_type == "rbf":
        if gamma is None:
            raise ValueError("gamma must be provided for RBF primal solver")
        return make_pipeline(
            Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=n_components,
                random_state=random_state,
            ),
            LinearSVC(
                C=C, dual=False, max_iter=50000, tol=1e-6, random_state=random_state
            ),
        )
    elif kernel_type == "poly":
        if gamma is None:
            raise ValueError("gamma must be provided for poly primal solver")
        return make_pipeline(
            Nystroem(
                kernel="poly",
                degree=3,
                coef0=1,
                gamma=gamma,
                n_components=n_components,
                random_state=random_state,
            ),
            LinearSVC(
                C=C, dual=False, max_iter=50000, tol=1e-6, random_state=random_state
            ),
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")


def compute_scale_gamma(X):
    """
    Compute gamma='scale' the same way sklearn does:
    gamma = 1 / (n_features * X.var())
    """
    return 1.0 / (X.shape[1] * X.var())


def get_dual_solver(kernel_type, C, random_state=42):
    """
    Return a dual solver for the given kernel.

    - Linear: LinearSVC(dual=True) or SVC(kernel='linear')
    - RBF: SVC(kernel='rbf')
    - Poly: SVC(kernel='poly')
    """
    if kernel_type == "linear":
        return LinearSVC(
            C=C, dual=True, max_iter=50000, tol=1e-6, random_state=random_state
        )
    elif kernel_type == "rbf":
        return SVC(C=C, kernel="rbf", gamma="scale", tol=1e-6, max_iter=50000)
    elif kernel_type == "poly":
        return SVC(
            C=C,
            kernel="poly",
            degree=3,
            coef0=1,
            gamma="scale",
            tol=1e-6,
            max_iter=50000,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")


# =========================================================
# Adaptive n_components finder
# =========================================================


def find_matched_n_components(
    kernel_type,
    C,
    X_train,
    y_train,
    X_test,
    y_test,
    dual_acc,
    acc_tol=0.03,
    candidates=None,
    random_state=42,
):
    """
    For nonlinear kernels, find the smallest n_components such that
    the primal approximation achieves accuracy within acc_tol of
    the dual's accuracy.

    Both RBF and poly use Nystroem, which is capped at n_train components.
    We always include n_train as a final candidate to use the maximum
    available approximation quality.

    Returns:
        (best_n_components, primal_acc_at_best)
    """
    if candidates is None:
        candidates = [100, 200, 500, 1000, 2000, 3000, 5000]

    # Nystroem (used for both rbf and poly) can't have more components
    # than training samples.
    n_train = X_train.shape[0]

    # Filter to feasible sizes and add n_train as the last resort
    candidates = [c for c in candidates if c <= n_train]
    if not candidates or candidates[-1] < n_train:
        candidates.append(n_train)

    gamma = compute_scale_gamma(X_train)

    for nc in candidates:
        model = get_primal_solver(kernel_type, C, nc, random_state, gamma=gamma)
        model.fit(X_train, y_train)
        p_acc = accuracy_score(y_test, model.predict(X_test))

        if abs(p_acc - dual_acc) <= acc_tol:
            return nc, p_acc

    # If none matched, return the largest we tried
    return candidates[-1], p_acc


# =========================================================
# Core experiment runner (shared by vary_n and vary_d)
# =========================================================


def _run_single_setting(
    kernel_type,
    C,
    X_train,
    X_test,
    y_train,
    y_test,
    n_samples,
    n_features,
    nd_ratio,
    experiment_label,
    acc_tol=0.03,
    n_timing_runs=5,
    random_state=42,
):
    """
    Run primal vs dual comparison for a single (kernel, dataset) setting.
    For nonlinear kernels, adaptively finds n_components first.

    The accuracy tolerance is adjusted for small test sets: with n_test=10,
    a single prediction changes accuracy by 0.1, so requiring 0.03 tolerance
    is impossible. We use max(acc_tol, 2/n_test) to account for this.
    """
    # Step 1: Fit dual and get reference accuracy
    dual_model = get_dual_solver(kernel_type, C, random_state)
    dual_model.fit(X_train, y_train)
    dual_acc = accuracy_score(y_test, dual_model.predict(X_test))

    # Adjust tolerance for small test sets
    n_test = len(y_test)
    effective_tol = max(acc_tol, 2.0 / n_test)

    # Step 2: Find appropriate n_components for primal (if nonlinear)
    n_comp_used = None
    if kernel_type == "linear":
        primal_model = get_primal_solver(kernel_type, C, random_state=random_state)
        primal_model.fit(X_train, y_train)
        primal_acc = accuracy_score(y_test, primal_model.predict(X_test))
        acc_matched = True
    else:
        n_comp_used, primal_acc = find_matched_n_components(
            kernel_type,
            C,
            X_train,
            y_train,
            X_test,
            y_test,
            dual_acc,
            acc_tol=effective_tol,
            random_state=random_state,
        )
        acc_matched = abs(primal_acc - dual_acc) <= effective_tol

    # Step 3: Time both solvers (using the matched n_components)
    gamma = compute_scale_gamma(X_train)
    if kernel_type == "linear":
        primal_for_timing = get_primal_solver(kernel_type, C, random_state=random_state)
    else:
        primal_for_timing = get_primal_solver(
            kernel_type, C, n_comp_used, random_state, gamma=gamma
        )

    primal_time = time_fit(primal_for_timing, X_train, y_train, n_runs=n_timing_runs)
    dual_time = time_fit(dual_model, X_train, y_train, n_runs=n_timing_runs)

    # Re-check accuracy after timing (since time_fit refits)
    primal_acc = accuracy_score(y_test, primal_for_timing.predict(X_test))
    dual_acc = accuracy_score(y_test, dual_model.predict(X_test))

    # Print progress
    comp_str = f", ncomp={n_comp_used}" if n_comp_used else ""
    tol_str = f", tol={effective_tol:.2f}" if effective_tol > acc_tol else ""
    match_str = "OK" if acc_matched else "UNMATCHED"
    print(
        f"primal={primal_time:.4f}s (acc={primal_acc:.3f}), "
        f"dual={dual_time:.4f}s (acc={dual_acc:.3f}) "
        f"[{match_str}{comp_str}{tol_str}]"
    )

    return {
        "kernel": kernel_type,
        "n_samples": n_samples,
        "n_features": n_features,
        "nd_ratio": nd_ratio,
        "primal_time": primal_time,
        "dual_time": dual_time,
        "primal_acc": primal_acc,
        "dual_acc": dual_acc,
        "n_components": n_comp_used,
        "acc_matched": acc_matched,
        "effective_tol": effective_tol,
        "experiment": experiment_label,
    }


# =========================================================
# Experiment 1: Vary n with fixed d
# =========================================================


def experiment_vary_n(
    n_samples_list,
    fixed_d,
    kernel_types,
    C=1.0,
    acc_tol=0.03,
    n_timing_runs=5,
    random_state=42,
    **kwargs,  # Accept and ignore legacy n_components param
):
    """
    Fix d, vary n. Measure primal vs dual training time.
    Uses adaptive n_components for nonlinear kernels.
    """
    results = []

    for kernel_type in kernel_types:
        print(f"\n--- Kernel: {kernel_type}, fixed d={fixed_d} ---")
        for n in n_samples_list:
            nd_ratio = n / fixed_d
            print(f"  n={n}, n/d={nd_ratio:.2f} ... ", end="", flush=True)

            X_train, X_test, y_train, y_test = generate_synthetic_data(
                n, fixed_d, noise=0.05, random_state=random_state
            )

            result = _run_single_setting(
                kernel_type,
                C,
                X_train,
                X_test,
                y_train,
                y_test,
                n_samples=n,
                n_features=fixed_d,
                nd_ratio=nd_ratio,
                experiment_label="vary_n",
                acc_tol=acc_tol,
                n_timing_runs=n_timing_runs,
                random_state=random_state,
            )
            results.append(result)

    return results


# =========================================================
# Experiment 2: Vary d with fixed n
# =========================================================


def experiment_vary_d(
    fixed_n,
    n_features_list,
    kernel_types,
    C=1.0,
    acc_tol=0.03,
    n_timing_runs=5,
    random_state=42,
    **kwargs,  # Accept and ignore legacy n_components param
):
    """
    Fix n, vary d. Measure primal vs dual training time.
    Uses adaptive n_components for nonlinear kernels.
    """
    results = []

    for kernel_type in kernel_types:
        print(f"\n--- Kernel: {kernel_type}, fixed n={fixed_n} ---")
        for d in n_features_list:
            nd_ratio = fixed_n / d
            print(f"  d={d}, n/d={nd_ratio:.2f} ... ", end="", flush=True)

            X_train, X_test, y_train, y_test = generate_synthetic_data(
                fixed_n, d, noise=0.05, random_state=random_state
            )

            result = _run_single_setting(
                kernel_type,
                C,
                X_train,
                X_test,
                y_train,
                y_test,
                n_samples=fixed_n,
                n_features=d,
                nd_ratio=nd_ratio,
                experiment_label="vary_d",
                acc_tol=acc_tol,
                n_timing_runs=n_timing_runs,
                random_state=random_state,
            )
            results.append(result)

    return results


# =========================================================
# Crossover estimation
# =========================================================


def estimate_crossover(results, kernel_type, experiment_type):
    """
    Estimate the n/d crossover point where primal becomes faster than dual
    using linear interpolation. Only uses accuracy-matched results for a
    fair comparison.

    Returns:
        dict with keys:
            crossovers: list of crossover n/d ratios
            confidence: 'high', 'moderate', 'low', or 'infeasible'
            n_matched: number of matched data points used
            n_total: total data points for this kernel/experiment
            n_unmatched: number of unmatched points excluded
            note: human-readable explanation
    """
    subset = [
        r
        for r in results
        if r["kernel"] == kernel_type and r["experiment"] == experiment_type
    ]
    subset.sort(key=lambda r: r["nd_ratio"])

    n_total = len(subset)
    matched = [r for r in subset if r.get("acc_matched", True)]
    n_matched = len(matched)
    n_unmatched = n_total - n_matched

    # If most points are unmatched, crossover estimation is unreliable
    if n_matched < 3:
        return {
            "crossovers": [],
            "confidence": "infeasible",
            "n_matched": n_matched,
            "n_total": n_total,
            "n_unmatched": n_unmatched,
            "note": "Too few accuracy-matched points for reliable estimation",
        }

    # Look for sign changes in (dual_time - primal_time) using matched results only
    diffs = [(r["nd_ratio"], r["dual_time"] - r["primal_time"]) for r in matched]

    crossovers = []
    for i in range(len(diffs) - 1):
        ratio1, diff1 = diffs[i]
        ratio2, diff2 = diffs[i + 1]
        if diff1 * diff2 < 0:  # sign change
            crossover = ratio1 + (ratio2 - ratio1) * abs(diff1) / (
                abs(diff1) + abs(diff2)
            )
            crossovers.append(crossover)

    # Assess confidence
    if not crossovers:
        # No crossover found — one solver dominates throughout matched range
        if matched:
            faster = (
                "Primal"
                if matched[0]["primal_time"] < matched[0]["dual_time"]
                else "Dual"
            )
            note = f"{faster} faster throughout matched range"
        else:
            note = "No data available"
        confidence = "high" if n_unmatched == 0 else "moderate"
    elif len(crossovers) == 1:
        if n_unmatched == 0:
            confidence = "high"
            note = f"Clean crossover at n/d ~ {crossovers[0]:.1f}"
        elif n_unmatched <= 2:
            confidence = "moderate"
            note = f"Crossover at n/d ~ {crossovers[0]:.1f} ({n_unmatched} unmatched points excluded)"
        else:
            confidence = "low"
            note = f"Crossover at n/d ~ {crossovers[0]:.1f} but {n_unmatched} points couldn't be matched"
    else:
        # Multiple crossovers = noisy, likely unreliable
        confidence = "low"
        note = (
            f"Multiple crossovers detected ({len(crossovers)}), "
            f"likely noise from approximation quality variation"
        )

    return {
        "crossovers": crossovers,
        "confidence": confidence,
        "n_matched": n_matched,
        "n_total": n_total,
        "n_unmatched": n_unmatched,
        "note": note,
    }


# =========================================================
# Plotting
# =========================================================


def plot_scaling_comparison(
    results, experiment_label, x_var, x_label, output_dir="plots"
):
    """
    Plot primal vs dual training time for each kernel.
    Shades regions where accuracy matching failed (unfair comparison).
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = sorted(
            [r for r in results if r["kernel"] == kernel], key=lambda r: r[x_var]
        )

        x_vals = [r[x_var] for r in subset]
        primal_times = [r["primal_time"] for r in subset]
        dual_times = [r["dual_time"] for r in subset]
        matched_flags = [r.get("acc_matched", True) for r in subset]

        # Plot all points but style matched vs unmatched differently
        matched_x = [x for x, m in zip(x_vals, matched_flags) if m]
        matched_pt = [t for t, m in zip(primal_times, matched_flags) if m]
        matched_dt = [t for t, m in zip(dual_times, matched_flags) if m]
        unmatched_x = [x for x, m in zip(x_vals, matched_flags) if not m]
        unmatched_pt = [t for t, m in zip(primal_times, matched_flags) if not m]
        unmatched_dt = [t for t, m in zip(dual_times, matched_flags) if not m]

        # Matched points: solid lines
        ax.plot(matched_x, matched_pt, "b-o", label="Primal", markersize=5)
        ax.plot(matched_x, matched_dt, "r-s", label="Dual", markersize=5)

        # Unmatched points: faded with X markers
        if unmatched_x:
            ax.plot(
                unmatched_x,
                unmatched_pt,
                "bx",
                markersize=8,
                alpha=0.4,
                markeredgewidth=2,
            )
            ax.plot(
                unmatched_x,
                unmatched_dt,
                "rx",
                markersize=8,
                alpha=0.4,
                markeredgewidth=2,
            )

            # Shade the unmatched region
            um_min = min(unmatched_x)
            um_max = max(unmatched_x)
            ax.axvspan(
                um_min * 0.8,
                um_max * 1.2,
                alpha=0.1,
                color="gray",
                label="Unmatched (unfair)",
            )

        # Mark crossover regions (only from matched data)
        info = estimate_crossover(results, kernel, experiment_label)
        for c in info["crossovers"]:
            ax.axvline(
                x=c,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Crossover ~ {c:.1f}",
            )

        # Add confidence annotation
        conf = info["confidence"]
        conf_color = {
            "high": "green",
            "moderate": "orange",
            "low": "red",
            "infeasible": "gray",
        }
        ax.annotate(
            f"Confidence: {conf}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=8,
            color=conf_color.get(conf, "black"),
            verticalalignment="top",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Training Time (seconds)")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Primal vs Dual Training Time ({experiment_label})", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"scaling_{experiment_label}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_accuracy_comparison(
    results, experiment_label, x_var, x_label, output_dir="plots"
):
    """
    Plot primal vs dual test accuracy to verify solution quality is comparable.
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = sorted(
            [r for r in results if r["kernel"] == kernel], key=lambda r: r[x_var]
        )

        x_vals = [r[x_var] for r in subset]
        primal_acc = [r["primal_acc"] for r in subset]
        dual_acc = [r["dual_acc"] for r in subset]

        ax.plot(x_vals, primal_acc, "b-o", label="Primal", markersize=5)
        ax.plot(x_vals, dual_acc, "r-s", label="Dual", markersize=5)

        ax.set_xlabel(x_label)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Solution Quality Check ({experiment_label})", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"accuracy_{experiment_label}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_crossover_summary(all_results, output_dir="plots"):
    """
    Summary plot showing estimated crossover points for each kernel,
    with confidence indicated by bar color.
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in all_results))
    experiments = sorted(set(r["experiment"] for r in all_results))

    conf_colors = {
        "high": "#2ecc71",
        "moderate": "#f39c12",
        "low": "#e74c3c",
        "infeasible": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    x_pos = np.arange(len(kernels))

    for exp_idx, exp in enumerate(experiments):
        crossover_vals = []
        bar_colors = []
        for kernel in kernels:
            info = estimate_crossover(all_results, kernel, exp)
            if info["crossovers"] and info["confidence"] != "infeasible":
                # Use first crossover if multiple (most reliable)
                crossover_vals.append(info["crossovers"][0])
            else:
                crossover_vals.append(0)
            bar_colors.append(conf_colors.get(info["confidence"], "#95a5a6"))

        bars = ax.bar(
            x_pos + exp_idx * bar_width,
            crossover_vals,
            bar_width,
            label=exp,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, val, kernel in zip(bars, crossover_vals, kernels):
            info = estimate_crossover(all_results, kernel, exp)
            if val > 0:
                label_text = f"{val:.1f}\n({info['confidence']})"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    label_text,
                    ha="center",
                    fontsize=8,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.5,
                    info["confidence"],
                    ha="center",
                    fontsize=7,
                    fontstyle="italic",
                    color="gray",
                )

    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("Crossover n/d Ratio")
    ax.set_title(
        "Estimated Crossover Points by Kernel and Experiment\n(color = confidence: green=high, orange=moderate, red=low, gray=infeasible)"
    )
    ax.set_xticks(x_pos + bar_width / 2)
    ax.set_xticklabels(kernels)
    ax.legend(title="Experiment")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "crossover_summary.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_ncomponents_used(
    results, experiment_label, x_var, x_label, output_dir="plots"
):
    """
    Plot the n_components required for accuracy matching at each setting.
    Only shows nonlinear kernels (linear doesn't use feature maps).
    """
    os.makedirs(output_dir, exist_ok=True)
    kernels = [k for k in sorted(set(r["kernel"] for r in results)) if k != "linear"]

    if not kernels:
        return

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = sorted(
            [r for r in results if r["kernel"] == kernel and r.get("n_components")],
            key=lambda r: r[x_var],
        )

        if not subset:
            continue

        x_vals = [r[x_var] for r in subset]
        nc_vals = [r["n_components"] for r in subset]
        matched = [r.get("acc_matched", True) for r in subset]

        # Color points by match status
        colors = ["green" if m else "red" for m in matched]
        ax.scatter(x_vals, nc_vals, c=colors, edgecolors="k", s=60, zorder=3)
        ax.plot(x_vals, nc_vals, "k--", alpha=0.3, zorder=1)

        ax.set_xlabel(x_label)
        ax.set_ylabel("n_components used")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")

        # Legend for match status
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markeredgecolor="k",
                markersize=8,
                label="Accuracy matched",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markeredgecolor="k",
                markersize=8,
                label="Unmatched",
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Adaptive n_components for Accuracy Matching ({experiment_label})", fontsize=14
    )
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"ncomponents_{experiment_label}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def print_scaling_table(results, label):
    """Print a timing summary table."""
    print(f"\n{'=' * 120}")
    print(f"SCALING RESULTS: {label}")
    print(f"{'=' * 120}")
    print(
        f"{'Kernel':<10} {'n':>6} {'d':>6} {'n/d':>8} "
        f"{'Primal(s)':>10} {'Dual(s)':>10} {'Faster':>8} "
        f"{'P_acc':>6} {'D_acc':>6} {'ncomp':>6} {'Match':>9}"
    )
    print("-" * 120)
    for r in results:
        faster = "Primal" if r["primal_time"] < r["dual_time"] else "Dual"
        nc = str(r.get("n_components", "-")) if r.get("n_components") else "-"
        matched = "OK" if r.get("acc_matched", True) else "UNMATCHED"
        print(
            f"{r['kernel']:<10} {r['n_samples']:>6} {r['n_features']:>6} "
            f"{r['nd_ratio']:>8.2f} "
            f"{r['primal_time']:>10.4f} {r['dual_time']:>10.4f} {faster:>8} "
            f"{r['primal_acc']:>6.3f} {r['dual_acc']:>6.3f} {nc:>6} {matched:>9}"
        )


if __name__ == "__main__":
    kernel_types = ["linear", "rbf", "poly"]
    C = 1.0

    print("=" * 60)
    print("RESEARCH QUESTION 2: SOLVER SCALING CROSSOVER")
    print("=" * 60)

    # ---- Experiment A: Fix d=100, vary n ----
    print("\n>>> Experiment A: Varying n with fixed d=100")
    fixed_d = 100
    n_samples_list = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    results_vary_n = experiment_vary_n(
        n_samples_list=n_samples_list,
        fixed_d=fixed_d,
        kernel_types=kernel_types,
        C=C,
        n_components=300,
        n_timing_runs=5,
    )

    print_scaling_table(results_vary_n, "Vary n (d=100)")

    # ---- Experiment B: Fix n=1000, vary d ----
    print("\n>>> Experiment B: Varying d with fixed n=1000")
    fixed_n = 1000
    n_features_list = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    results_vary_d = experiment_vary_d(
        fixed_n=fixed_n,
        n_features_list=n_features_list,
        kernel_types=kernel_types,
        C=C,
        n_components=300,
        n_timing_runs=5,
    )

    print_scaling_table(results_vary_d, "Vary d (n=1000)")

    # ---- Combined results ----
    all_results = results_vary_n + results_vary_d

    # ---- Plots ----
    plot_scaling_comparison(
        results_vary_n, "vary_n", "n_samples", "Number of Samples (n)"
    )
    plot_scaling_comparison(
        results_vary_d, "vary_d", "n_features", "Number of Features (d)"
    )
    plot_accuracy_comparison(
        results_vary_n, "vary_n", "n_samples", "Number of Samples (n)"
    )
    plot_accuracy_comparison(
        results_vary_d, "vary_d", "n_features", "Number of Features (d)"
    )
    plot_crossover_summary(all_results)

    # ---- Report crossover points ----
    print("\n" + "=" * 60)
    print("ESTIMATED CROSSOVER POINTS")
    print("=" * 60)
    for kernel in kernel_types:
        for exp in ["vary_n", "vary_d"]:
            crossovers = estimate_crossover(all_results, kernel, exp)
            if crossovers:
                print(
                    f"  {kernel} ({exp}): n/d crossover at {[f'{c:.2f}' for c in crossovers]}"
                )
            else:
                direction = (
                    "Primal"
                    if all_results[0]["primal_time"] < all_results[0]["dual_time"]
                    else "Dual"
                )
                print(
                    f"  {kernel} ({exp}): No crossover detected — {direction} faster throughout"
                )

    print("\nDone! All plots saved to ./plots/")
