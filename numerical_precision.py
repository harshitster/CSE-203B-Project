import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from itertools import product
import warnings
import os

from utils import (
    generate_synthetic_data,
    compute_primal_objective,
    compute_dual_objective,
    check_kkt_conditions,
)

warnings.filterwarnings("ignore")


def extract_primal_dual_from_svc(
    model, X_train, y_train, C, kernel_type, kernel_params
):
    alpha_full = np.zeros(len(y_train))
    sv_indices = model.support_
    alpha_sv = np.abs(model.dual_coef_[0])
    alpha_full[sv_indices] = alpha_sv

    if kernel_type == "linear":
        K = linear_kernel(X_train)
    elif kernel_type == "rbf":
        gamma = kernel_params.get("gamma", "scale")
        if gamma == "scale":
            gamma = 1.0 / (X_train.shape[1] * X_train.var())
        K = rbf_kernel(X_train, gamma=gamma)
    elif kernel_type == "poly":
        degree = kernel_params.get("degree", 3)
        coef0 = kernel_params.get("coef0", 1)
        gamma = kernel_params.get("gamma", "scale")
        if gamma == "scale":
            gamma = 1.0 / (X_train.shape[1] * X_train.var())
        K = polynomial_kernel(X_train, degree=degree, coef0=coef0, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel: {kernel_type}")

    b = model.intercept_[0]

    dual_obj = compute_dual_objective(alpha_full, y_train, K)

    ay = alpha_full * y_train
    w_norm_sq = ay @ K @ ay
    decision = K @ ay + b
    margins = y_train * decision
    hinge_losses = np.maximum(0, 1 - margins)
    primal_obj = 0.5 * w_norm_sq + C * np.sum(hinge_losses)

    kkt_stats = check_kkt_conditions(alpha_full, y_train, K, b, C)

    n_support_vectors = len(sv_indices)

    return {
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "gap": abs(primal_obj - dual_obj),
        "relative_gap": abs(primal_obj - dual_obj) / max(abs(primal_obj), 1e-10),
        "n_support_vectors": n_support_vectors,
        "kkt_stats": kkt_stats,
    }


def run_experiment(
    n_samples_list,
    n_features_list,
    noise_list,
    C_list,
    kernel_configs,
    random_state=42,
):
    results = []
    total = (
        len(n_samples_list)
        * len(n_features_list)
        * len(noise_list)
        * len(C_list)
        * len(kernel_configs)
    )
    count = 0

    for n_samples, n_features, noise, C_val, (kernel_type, kernel_params) in product(
        n_samples_list, n_features_list, noise_list, C_list, kernel_configs
    ):
        count += 1
        print(
            f"[{count}/{total}] n={n_samples}, d={n_features}, "
            f"noise={noise}, C={C_val}, kernel={kernel_type}"
        )

        try:
            X_train, X_test, y_train, y_test = generate_synthetic_data(
                n_samples, n_features, noise, random_state
            )

            svc_params = {
                "C": C_val,
                "kernel": kernel_type,
                "tol": 1e-6,
                "max_iter": 50000,
            }
            if kernel_type == "rbf":
                gamma = kernel_params.get("gamma", "scale")
                svc_params["gamma"] = gamma
            elif kernel_type == "poly":
                svc_params["degree"] = kernel_params.get("degree", 3)
                svc_params["coef0"] = kernel_params.get("coef0", 1)
                gamma = kernel_params.get("gamma", "scale")
                svc_params["gamma"] = gamma

            model = SVC(**svc_params)
            model.fit(X_train, y_train)

            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            info = extract_primal_dual_from_svc(
                model, X_train, y_train, C_val, kernel_type, kernel_params
            )

            results.append(
                {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "noise": noise,
                    "C": C_val,
                    "kernel": kernel_type,
                    "primal_obj": info["primal_obj"],
                    "dual_obj": info["dual_obj"],
                    "gap": info["gap"],
                    "relative_gap": info["relative_gap"],
                    "n_support_vectors": info["n_support_vectors"],
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "cs_violations": info["kkt_stats"][
                        "complementary_slackness_violations"
                    ],
                    "df_violations": info["kkt_stats"]["dual_feasibility_violations"],
                    "max_cs_violation": info["kkt_stats"]["max_cs_violation"],
                    "max_df_violation": info["kkt_stats"]["max_df_violation"],
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results


def plot_gap_vs_variable(results, variable, variable_label, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = [r for r in results if r["kernel"] == kernel]

        x_vals = [r[variable] for r in subset]
        y_vals = [r["relative_gap"] for r in subset]

        ax.scatter(x_vals, y_vals, alpha=0.6, edgecolors="k", linewidth=0.5)
        ax.set_xlabel(variable_label)
        ax.set_ylabel("Relative Primal-Dual Gap")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Relative Primal-Dual Gap vs {variable_label}", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"gap_vs_{variable}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_kkt_violations(results, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))
    C_values = sorted(set(r["C"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        for C_val in C_values:
            subset = [r for r in results if r["kernel"] == kernel and r["C"] == C_val]
            if not subset:
                continue
            n_samples_vals = [r["n_samples"] for r in subset]
            cs_vals = [r["max_cs_violation"] for r in subset]
            ax.plot(n_samples_vals, cs_vals, marker="o", label=f"C={C_val}", alpha=0.7)

        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Max Complementary Slackness Violation")
        ax.set_title(f"Kernel: {kernel}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("KKT Complementary Slackness Violations", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "kkt_violations.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def plot_gap_heatmap(results, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    kernels = sorted(set(r["kernel"] for r in results))

    fig, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), squeeze=False
    )

    for idx, kernel in enumerate(kernels):
        ax = axes[0][idx]
        subset = [r for r in results if r["kernel"] == kernel]

        n_vals = sorted(set(r["n_samples"] for r in subset))
        c_vals = sorted(set(r["C"] for r in subset))

        gap_matrix = np.full((len(c_vals), len(n_vals)), np.nan)
        for r in subset:
            i = c_vals.index(r["C"])
            j = n_vals.index(r["n_samples"])
            if np.isnan(gap_matrix[i, j]):
                gap_matrix[i, j] = r["relative_gap"]
            else:
                gap_matrix[i, j] = (gap_matrix[i, j] + r["relative_gap"]) / 2

        im = ax.imshow(
            np.log10(gap_matrix + 1e-16),
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
        )
        ax.set_xticks(range(len(n_vals)))
        ax.set_xticklabels(n_vals, rotation=45)
        ax.set_yticks(range(len(c_vals)))
        ax.set_yticklabels(c_vals)
        ax.set_xlabel("n_samples")
        ax.set_ylabel("C")
        ax.set_title(f"Kernel: {kernel}")
        plt.colorbar(im, ax=ax, label="log10(relative gap)")

    plt.suptitle("Relative Primal-Dual Gap Heatmap", fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "gap_heatmap.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def print_summary_table(results):
    print("\n" + "=" * 120)
    print(
        f"{'Kernel':<10} {'n':>6} {'d':>6} {'noise':>6} {'C':>8} "
        f"{'Primal':>12} {'Dual':>12} {'Gap':>12} {'RelGap':>12} "
        f"{'SVs':>5} {'KKT_viol':>9}"
    )
    print("=" * 120)

    for r in results:
        print(
            f"{r['kernel']:<10} {r['n_samples']:>6} {r['n_features']:>6} "
            f"{r['noise']:>6.2f} {r['C']:>8.2f} "
            f"{r['primal_obj']:>12.4f} {r['dual_obj']:>12.4f} "
            f"{r['gap']:>12.6f} {r['relative_gap']:>12.2e} "
            f"{r['n_support_vectors']:>5} {r['cs_violations']:>9}"
        )


if __name__ == "__main__":
    n_samples_list = [100, 500, 1000, 3000]
    n_features_list = [10, 50, 200]
    noise_list = [0.0, 0.1, 0.3]
    C_list = [0.01, 1.0, 100.0, 10000.0]

    kernel_configs = [
        ("linear", {}),
        ("rbf", {"gamma": "scale"}),
        ("poly", {"degree": 3, "coef0": 1, "gamma": "scale"}),
    ]

    print("=" * 60)
    print("RESEARCH QUESTION 1: NUMERICAL PRECISION STUDY")
    print("=" * 60)

    results = run_experiment(
        n_samples_list=n_samples_list,
        n_features_list=n_features_list,
        noise_list=noise_list,
        C_list=C_list,
        kernel_configs=kernel_configs,
    )

    print_summary_table(results)

    plot_gap_vs_variable(results, "n_samples", "Number of Samples")
    plot_gap_vs_variable(results, "n_features", "Number of Features")
    plot_gap_vs_variable(results, "C", "Regularization (C)")
    plot_gap_vs_variable(results, "noise", "Label Noise")
    plot_kkt_violations(results)
    plot_gap_heatmap(results)

    print("\nDone! All plots saved to ./plots/")
