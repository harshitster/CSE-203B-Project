import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples, n_features, noise=0.0, random_state=42):
    """
    Generate a binary classification dataset with controlled dimensions and noise.
    
    Parameters:
        n_samples: number of data points
        n_features: number of features
        noise: fraction of labels to flip (0.0 = clean, 0.3 = 30% noisy)
        random_state: seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test (scaled)
    """
    n_informative = max(2, n_features // 2)
    n_redundant = max(0, n_features - n_informative - min(2, n_features // 4))
    n_clusters_per_class = 1

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        flip_y=noise,
        class_sep=1.0,
        random_state=random_state,
    )

    # Convert labels to {-1, +1}
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def time_fit(model, X, y, n_runs=5):
    """
    Time the .fit() call of a model, returning median wall-clock seconds.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.fit(X, y)
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times)


def compute_primal_objective(w, b, X, y, C):
    """
    Compute the primal SVM objective: 0.5 * ||w||^2 + C * sum(max(0, 1 - y_i(w.x_i + b)))
    """
    margins = y * (X @ w + b)
    hinge_losses = np.maximum(0, 1 - margins)
    return 0.5 * np.dot(w, w) + C * np.sum(hinge_losses)


def compute_dual_objective(alpha, y, K):
    """
    Compute the dual SVM objective: sum(alpha) - 0.5 * alpha^T (y y^T * K) alpha
    """
    ay = alpha * y
    return np.sum(alpha) - 0.5 * ay @ K @ ay


def check_kkt_conditions(alpha, y, K, b, C, tol=1e-4):
    """
    Check KKT conditions and return violation statistics.
    
    Returns:
        dict with complementary_slackness_violations, dual_feasibility_violations,
        max_cs_violation, max_df_violation
    """
    n = len(y)
    decision = K @ (alpha * y) + b
    margins = y * decision

    cs_violations = 0
    max_cs_violation = 0.0
    df_violations = 0
    max_df_violation = 0.0

    for i in range(n):
        # Dual feasibility: 0 <= alpha_i <= C
        if alpha[i] < -tol or alpha[i] > C + tol:
            df_violations += 1
            max_df_violation = max(max_df_violation, max(-alpha[i], alpha[i] - C))

        # Complementary slackness conditions:
        # alpha_i = 0 => margin_i >= 1
        # 0 < alpha_i < C => margin_i = 1
        # alpha_i = C => margin_i <= 1
        if alpha[i] < tol:
            if margins[i] < 1 - tol:
                cs_violations += 1
                max_cs_violation = max(max_cs_violation, 1 - margins[i])
        elif alpha[i] > C - tol:
            if margins[i] > 1 + tol:
                cs_violations += 1
                max_cs_violation = max(max_cs_violation, margins[i] - 1)
        else:
            if abs(margins[i] - 1) > tol:
                cs_violations += 1
                max_cs_violation = max(max_cs_violation, abs(margins[i] - 1))

    return {
        "complementary_slackness_violations": cs_violations,
        "dual_feasibility_violations": df_violations,
        "max_cs_violation": max_cs_violation,
        "max_df_violation": max_df_violation,
        "total_samples": n,
    }
