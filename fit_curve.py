#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --- Helpers --- #
def fit_centerline(points_xy):
    # Building a matrix for linear regression y = slope * x + intercept
    design = np.vstack([points_xy[:, 0], np.ones(len(points_xy))]).T
    slope, intercept = np.linalg.lstsq(design, points_xy[:, 1], rcond=None)[0]
    theta = math.atan(slope)

    # slope near zero would cause division by near zero when computing X, so error occurs
    if abs(slope) < 1e-9:
        raise ValueError("Slope ~0; cannot derive X reliably.")

    x_offset = (42.0 - intercept) / slope
    return theta, x_offset, slope, intercept


def compute_frame(points_xy, theta, x_offset):
    origin = np.array([x_offset, 42.0])
    u = np.array([math.cos(theta), math.sin(theta)])    # unit vector along the line
    n = np.array([-math.sin(theta), math.cos(theta)])   # unit vector perpendicular to the line

    shifted = points_xy - origin
    t_vals = shifted @ u
    perp_vals = shifted @ n
    sin_term = np.sin(0.3 * t_vals)
    return t_vals, perp_vals, sin_term


def mse_for_M(M, t_vals, perp_vals, sin_term):
    pred = np.exp(M * np.abs(t_vals)) * sin_term
    return np.mean((perp_vals - pred) ** 2)


def best_M_1d(t_vals, perp_vals, sin_term, low=-0.05, high=0.05, iters=40):
    lo, hi = low, high
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = mse_for_M(m1, t_vals, perp_vals, sin_term)
        f2 = mse_for_M(m2, t_vals, perp_vals, sin_term)
        if f1 < f2:
            hi = m2
        else:
            lo = m1

    M = 0.5 * (lo + hi)
    err = mse_for_M(M, t_vals, perp_vals, sin_term)
    return M, err


def refine_theta_X(points_xy, theta0, X0, max_outer=40):
    def clamp_theta(th):
        eps = 1e-6
        lo, hi = eps, math.radians(50) - eps
        return min(max(th, lo), hi)

    def clamp_X(X):
        return float(min(max(X, 0.0 + 1e-6), 100.0 - 1e-6))

    theta = clamp_theta(theta0)
    x_offset = clamp_X(X0)

    # initialize M using current theta/X
    t_vals, perp_vals, sin_term = compute_frame(points_xy, theta, x_offset)
    M, best_err = best_M_1d(t_vals, perp_vals, sin_term)

    step_th = 1.0 * math.pi / 180.0 
    step_X = 0.5

    for _ in range(max_outer):
        improved = False
        for dth in (0.0, +step_th, -step_th):
            for dX in (0.0, +step_X, -step_X):
                th = clamp_theta(theta + dth)
                Xc = clamp_X(x_offset + dX)

                t_vals_c, perp_vals_c, sin_term_c = compute_frame(points_xy, th, Xc)
                M_c, err_c = best_M_1d(t_vals_c, perp_vals_c, sin_term_c)

                if err_c + 1e-12 < best_err:
                    theta, x_offset, M, best_err = th, Xc, M_c, err_c
                    improved = True

        if not improved:
            step_th *= 0.5
            step_X *= 0.5
        if step_th < (0.01 * math.pi / 180.0) and step_X < 1e-3:
            break

    return theta, x_offset, M, best_err


def sample_curve(theta, M, x_offset, n=800):
    t = np.linspace(6.001, 59.999, n)
    ct, st = math.cos(theta), math.sin(theta)
    amp = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * ct - amp * st + x_offset
    y = 42.0 + t * st + amp * ct
    return np.stack([x, y], axis=1)


def l1_set_distance(curve_pts, data_pts):
    # Brute-force pairwise L1 (n_curve x n_data); fine for this size. or else we can you other methods like euclidean, k-d trees etc.
    diff = curve_pts[:, None, :] - data_pts[None, :, :]
    dists = np.abs(diff).sum(axis=2)  # L1 norm
    nearest = dists.min(axis=1)
    return dict(
        mean=float(nearest.mean()),
        median=float(np.median(nearest)),
        p90=float(np.quantile(nearest, 0.90)),
        p99=float(np.quantile(nearest, 0.99)),
        max=float(nearest.max()),
    )


def desmos_string(theta, M, Xoff):
    return (
        r"\left(t*\cos(" + f"{theta:.6f}" + r")-e^{" + f"{M:.6f}" + r"\left|t\right|}"
        + r"\cdot\sin(0.3t)\sin(" + f"{theta:.6f}" + r")+" + f"{Xoff:.6f}"
        + r",\ 42+t*\sin(" + f"{theta:.6f}" + r")+e^{" + f"{M:.6f}"
        + r"\left|t\right|}\cdot\sin(0.3t)\cos(" + f"{theta:.6f}" + r")\right)"
    )


# ------------------------------ Main ------------------------------ #
def main():
    data_path = Path("XY_DATA.csv")
    if not data_path.exists():
        raise FileNotFoundError("XY_DATA.csv not found in working directory.")

    df = pd.read_csv(data_path)
    pts = df[["x", "y"]].to_numpy()

    # 1) initial line
    theta0, X0, s, c = fit_centerline(pts)

    # Clamp initial guesses into bounds
    theta0 = min(max(theta0, 1e-6), math.radians(50) - 1e-6)
    X0 = float(min(max(X0, 1e-6), 100.0 - 1e-6))

    # 2) refine (theta, X) with best-M each step
    theta, Xoff, M, err = refine_theta_X(pts, theta0, X0)

    # 3) final report of parameter values
    theta_deg = math.degrees(theta)
    print("\n=== Fitted Parameters ===")
    print(f"theta (rad)  : {theta:.9f}")
    print(f"theta (deg)  : {theta_deg:.6f}")
    print(f"M            : {M:.9f}")
    print(f"X            : {Xoff:.9f}")
    print(f"refit MSE    : {err:.9e}")

    # 4) Desmos equation generation
    desmos = desmos_string(theta, M, Xoff)
    print("\nDesmos/LaTeX:")
    print(desmos)

    # 5) L1 set distance (curve→data)
    curve = sample_curve(theta, M, Xoff, n=800)
    metrics = l1_set_distance(curve, pts)
    print("\nSet-to-set L1 distance (curve→nearest data):")
    for k, v in metrics.items():
        print(f"  {k:>6}: {v:.6f}")

    # 6) plot and save
    # Line without wave:
    t = np.linspace(6.001, 59.999, 600)
    x_line = t * math.cos(theta) + Xoff
    y_line = 42.0 + t * math.sin(theta)

    plt.figure(figsize=(8, 5))
    plt.plot(curve[:, 0], curve[:, 1], label="Fitted Curve", linewidth=2)
    plt.plot(x_line, y_line, "--", label="Centerline", linewidth=1)
    plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.25, label="Data (CSV)")
    plt.title("Parametric Curve Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=180)
    print("\nSaved plot -> visualization.png")


if __name__ == "__main__":
    main()