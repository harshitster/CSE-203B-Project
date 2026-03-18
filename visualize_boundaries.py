import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
import os


def plot_decision_boundary(ax, model, X, y, title=""):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    ax.contour(xx, yy, Z, colors="k", linewidths=0.5)

    try:
        Z_dec = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_dec = Z_dec.reshape(xx.shape)
        ax.contour(
            xx,
            yy,
            Z_dec,
            levels=[-1, 0, 1],
            colors=["blue", "black", "red"],
            linestyles=["--", "-", "--"],
            linewidths=[1, 2, 1],
        )
    except Exception:
        pass

    ax.scatter(
        X[y == -1, 0], X[y == -1, 1], c="blue", edgecolors="k", s=30, label="Class -1"
    )
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1], c="red", edgecolors="k", s=30, label="Class +1"
    )

    if hasattr(model, "support_"):
        ax.scatter(
            X[model.support_, 0],
            X[model.support_, 1],
            s=100,
            facecolors="none",
            edgecolors="green",
            linewidths=2,
            label="Support Vectors",
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def generate_2d_datasets():
    datasets = {}

    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42,
    )
    y = 2 * y - 1
    scaler = StandardScaler()
    datasets["Linear"] = (scaler.fit_transform(X), y)

    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    y = 2 * y - 1
    scaler = StandardScaler()
    datasets["Moons"] = (scaler.fit_transform(X), y)

    X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
    y = 2 * y - 1
    scaler = StandardScaler()
    datasets["Circles"] = (scaler.fit_transform(X), y)

    return datasets


def visualize_all(C=1.0, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    datasets = generate_2d_datasets()

    kernel_configs = [
        ("linear", {}),
        ("rbf", {"gamma": "scale"}),
        ("poly", {"degree": 3, "coef0": 1, "gamma": "scale"}),
    ]

    for dataset_name, (X, y) in datasets.items():
        fig, axes = plt.subplots(
            len(kernel_configs), 2, figsize=(12, 5 * len(kernel_configs))
        )

        for k_idx, (kernel_type, kernel_params) in enumerate(kernel_configs):
            svc_params = {"C": C, "kernel": kernel_type, "tol": 1e-6}
            if kernel_type == "rbf":
                svc_params["gamma"] = kernel_params.get("gamma", "scale")
            elif kernel_type == "poly":
                svc_params["degree"] = kernel_params.get("degree", 3)
                svc_params["coef0"] = kernel_params.get("coef0", 1)
                svc_params["gamma"] = kernel_params.get("gamma", "scale")

            dual_model = SVC(**svc_params)
            dual_model.fit(X, y)
            plot_decision_boundary(
                axes[k_idx, 0], dual_model, X, y, title=f"Dual — {kernel_type} kernel"
            )

            n_components = min(500, X.shape[0])
            gamma_val = 1.0 / (X.shape[1] * X.var())
            if kernel_type == "linear":
                primal_model = LinearSVC(C=C, dual=False, max_iter=50000, tol=1e-6)
            elif kernel_type == "rbf":
                primal_model = make_pipeline(
                    Nystroem(
                        kernel="rbf",
                        gamma=gamma_val,
                        n_components=n_components,
                        random_state=42,
                    ),
                    LinearSVC(C=C, dual=False, max_iter=50000, tol=1e-6),
                )
            elif kernel_type == "poly":
                primal_model = make_pipeline(
                    Nystroem(
                        kernel="poly",
                        degree=3,
                        coef0=1,
                        gamma=gamma_val,
                        n_components=n_components,
                        random_state=42,
                    ),
                    LinearSVC(C=C, dual=False, max_iter=50000, tol=1e-6),
                )

            primal_model.fit(X, y)
            plot_decision_boundary(
                axes[k_idx, 1],
                primal_model,
                X,
                y,
                title=f"Primal — {kernel_type} kernel",
            )

        axes[0, 0].set_ylabel("linear", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("rbf", fontsize=12, fontweight="bold")
        axes[2, 0].set_ylabel("poly", fontsize=12, fontweight="bold")

        fig.suptitle(
            f"Decision Boundaries: {dataset_name} Dataset",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        filepath = os.path.join(output_dir, f"boundaries_{dataset_name.lower()}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    print("=" * 60)
    print("DECISION BOUNDARY VISUALIZATION")
    print("=" * 60)
    visualize_all(C=1.0)
    print("\nDone! All boundary plots saved to ./plots/")
