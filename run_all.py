import sys
import os


def run_rq1():
    print("\n" + "#" * 60)
    print("# RESEARCH QUESTION 1: NUMERICAL PRECISION")
    print("#" * 60 + "\n")

    from numerical_precision import (
        run_experiment,
        print_summary_table,
        plot_gap_vs_variable,
        plot_kkt_violations,
        plot_gap_heatmap,
    )

    n_samples_list = [100, 500, 1000, 3000]
    n_features_list = [10, 50, 200]
    noise_list = [0.0, 0.1, 0.3]
    C_list = [0.01, 1.0, 100.0, 10000.0]

    kernel_configs = [
        ("linear", {}),
        ("rbf", {"gamma": "scale"}),
        ("poly", {"degree": 3, "coef0": 1, "gamma": "scale"}),
    ]

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

    print("\nRQ1 Done! Plots saved to ./plots/")


def run_rq2():
    print("\n" + "#" * 60)
    print("# RESEARCH QUESTION 2: SOLVER SCALING CROSSOVER")
    print("#" * 60 + "\n")

    from solver_scaling import (
        experiment_vary_n,
        experiment_vary_d,
        print_scaling_table,
        plot_scaling_comparison,
        plot_accuracy_comparison,
        plot_crossover_summary,
        plot_ncomponents_used,
        estimate_crossover,
    )

    kernel_types = ["linear", "rbf", "poly"]
    C = 1.0

    print(">>> Experiment A: Varying n with fixed d=100")
    fixed_d = 100
    n_samples_list = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    results_vary_n = experiment_vary_n(
        n_samples_list=n_samples_list,
        fixed_d=fixed_d,
        kernel_types=kernel_types,
        C=C,
        acc_tol=0.03,
        n_timing_runs=5,
    )
    print_scaling_table(results_vary_n, "Vary n (d=100)")

    print("\n>>> Experiment B: Varying d with fixed n=1000")
    fixed_n = 1000
    n_features_list = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    results_vary_d = experiment_vary_d(
        fixed_n=fixed_n,
        n_features_list=n_features_list,
        kernel_types=kernel_types,
        C=C,
        acc_tol=0.03,
        n_timing_runs=5,
    )
    print_scaling_table(results_vary_d, "Vary d (n=1000)")

    all_results = results_vary_n + results_vary_d

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
    plot_ncomponents_used(
        results_vary_n, "vary_n", "n_samples", "Number of Samples (n)"
    )
    plot_ncomponents_used(
        results_vary_d, "vary_d", "n_features", "Number of Features (d)"
    )
    plot_crossover_summary(all_results)

    print("\n" + "=" * 80)
    print("ESTIMATED CROSSOVER POINTS")
    print("=" * 80)
    for kernel in kernel_types:
        print(f"\n  {kernel.upper()} kernel:")
        for exp in ["vary_n", "vary_d"]:
            info = estimate_crossover(all_results, kernel, exp)
            conf = info["confidence"].upper()
            matched_str = f"{info['n_matched']}/{info['n_total']} matched"
            if info["crossovers"] and info["confidence"] != "infeasible":
                vals = [f"{c:.1f}" for c in info["crossovers"]]
                print(f"    {exp}: n/d ~ {', '.join(vals)}  [{conf}, {matched_str}]")
            else:
                print(f"    {exp}: {info['note']}  [{conf}, {matched_str}]")

    print("\n" + "-" * 80)
    print("INTERPRETATION GUIDE:")
    print("  HIGH     = clean single crossover, all points accuracy-matched")
    print("  MODERATE = crossover found but some points excluded or minor noise")
    print("  LOW      = multiple crossovers (noisy) or many unmatched points")
    print("  INFEASIBLE = primal approximation could not match dual accuracy")
    print("-" * 80)

    print("\nRQ2 Done! Plots saved to ./plots/")


def run_cvxpy():
    print("\n" + "#" * 60)
    print("# CVXPY CONTROL EXPERIMENT")
    print("#" * 60 + "\n")

    from cvxpy_control import (
        run_control_experiment,
        print_control_table,
        plot_control_scaling,
        plot_control_duality_gap,
    )

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
    plot_control_scaling(results)
    plot_control_duality_gap(results)

    print("\nCVXPY control experiment done! Plots saved to ./plots/")


def run_rq2_with_control():
    run_rq2()

    print("\n" + "#" * 60)
    print("# CVXPY CONTROL + DIRECTION COMPARISON")
    print("#" * 60 + "\n")

    from cvxpy_control import (
        run_control_experiment,
        print_control_table,
        print_direction_comparison,
        plot_control_scaling,
        plot_control_duality_gap,
        plot_sklearn_vs_cvxpy_direction,
    )
    from solver_scaling import experiment_vary_n, experiment_vary_d

    n_samples_list = [50, 100, 200, 500]
    n_features_list = [10, 50, 200, 500]
    kernel_types = ["linear", "rbf", "poly"]
    C = 1.0

    cvxpy_results = run_control_experiment(
        n_samples_list=n_samples_list,
        n_features_list=n_features_list,
        kernel_types=kernel_types,
        C=C,
        solver="SCS",
        n_runs=3,
    )

    print_control_table(cvxpy_results)
    plot_control_scaling(cvxpy_results)
    plot_control_duality_gap(cvxpy_results)

    print("\n>>> Running sklearn on matching settings for comparison...")
    sklearn_results = []
    for n in n_samples_list:
        for d in n_features_list:
            results_nd = experiment_vary_n(
                n_samples_list=[n],
                fixed_d=d,
                kernel_types=kernel_types,
                C=C,
                acc_tol=0.03,
                n_timing_runs=3,
            )
            sklearn_results.extend(results_nd)

    print_direction_comparison(cvxpy_results, sklearn_results)
    plot_sklearn_vs_cvxpy_direction(cvxpy_results, sklearn_results)


def run_viz():
    print("\n" + "#" * 60)
    print("# DECISION BOUNDARY VISUALIZATION")
    print("#" * 60 + "\n")

    from visualize_boundaries import visualize_all

    visualize_all(C=1.0)

    print("\nVisualization Done! Plots saved to ./plots/")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)

    if len(sys.argv) > 1:
        task = sys.argv[1].lower()
        if task == "rq1":
            run_rq1()
        elif task == "rq2":
            run_rq2()
        elif task == "cvxpy":
            run_cvxpy()
        elif task == "rq2full":
            run_rq2_with_control()
        elif task == "viz":
            run_viz()
        else:
            print(f"Unknown task: {task}")
            print("Usage: python run_all.py [rq1 | rq2 | cvxpy | rq2full | viz]")
    else:
        run_rq1()
        run_rq2()
        run_cvxpy()
        run_viz()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("Plots saved to ./plots/")
    print("=" * 60)
