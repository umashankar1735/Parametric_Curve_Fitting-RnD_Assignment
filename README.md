# Assignment for Research and Development / AI - Curve Fitting Problem

This repository contains code and visualizations for fitting the parametric curve

$$
\begin{aligned}
x(t) &= t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X \\
y(t) &= 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)
\end{aligned}
$$

with unknowns: \(\theta\) (radians), \(M\), and \(X\). Bounds used in the fit:

- \(0^\circ < \theta < 50^\circ\)
- \(-0.05 < M < 0.05\)
- \(0 < X < 100\)
- parameter t range: \(6 < t < 60\)

---
## What does the curve means(my understanding)


## Contents

- `fit_curve.py`  — main estimator script (reads `XY_DATA.csv`, fits parameters, prints results, saves `visualization.png`).
- `XY_DATA.csv`   — provided data (must be placed in the same folder before running).
- `visualization.png` — generated plot after a successful run.
- `Visualization-Desmos.png` - this is a visualization sucessfully run in desmos.
- `README.md`     — it has step by step solution of the problem and all the information of this research.
- `requirements.txt` — Python dependencies for reproducible runs.
- `Final-Results.txt` — All the result value M,X,theta & L1 mean-distance are stored in this file for a final check
---

## Detailed Step-by-step explanation

Below is a concise, stepwise description of the problem, the mathematical
model, and how the code implements the estimator.

1) Problem statement and model
	 - We are given 2D points (x, y) that lie on a curve parameterized by t
		 (unknown perpendicular-point parameter) and three global unknowns: \(\theta\),
		 \(M\), and \(X\). 
         The parametric equations are:
		 x(t) = t*cos(θ) - e^{M|t|} * sin(0.3 t) * sin(θ) + X
		 y(t) = 42 + t*sin(θ) + e^{M|t|} * sin(0.3 t) * cos(θ)

	 - Goal: estimate θ (orientation), M (amplitude growth), and X
		 (global x offset) using the provided (x,y) samples which are in "xy_data.csv" for t in (6,60).

2) Parameter bounds
	 - θ in (0°, 50°) — implemented in radians inside the code.
	 - M in (-0.05, 0.05).
	 - X in (0, 100).

3) High-level strategy (why this decomposition works)
	 - The parametrization can be written as a straight centerline (a line with
		 angle θ) plus a perpendicular oscillatory offset whose amplitude is
		 a function exp(M|t|) times sin(0.3 t). If we find θ and X, the data can
		 be projected into a coordinate frame where the perpendicular offsets
		 (d values) are a one-dimensional amplitude-only signal to fit for M.

4) Step A — fit a centerline to initialize θ and X
	 - Fit a simple linear model y ≈ slope * x + intercept by least-squares.
	 - Convert slope(=tanθ) to θ = arctan(slope). 
         Use intercept(X,42) to produce an initial X: X = (42 - intercept) / slope. 
         This gives a robust starting point for the nonlinear refinement.

5) Step B — rotate/translate points into the model frame
	 - Define base_point(b) = (X, 42). Define two orthonormal vectors:
		 u = (cos θ, sin θ) (along the centerline) and 
         n = (-sin θ, cos θ)(perpendicular).
	 - For each data point, compute t_i = (p_i - b)·u and d_i = (p_i - b)·n.
	 - The model predicts d_i ≈ exp(M * |t_i|) * sin(0.3 * t_i).

6) Step C — estimate M for a fixed (θ, X)
	 - With t_i, d_i and sin(0.3 t_i) known, M is a single scalar. 
        We search for M in the allowed interval and evaluate mean-squared error
		mse(M) = mean((d_i - exp(M|t_i|) sin(0.3 t_i))^2).
	 - The code uses a ternary-like bracketed search over (-0.05,0.05) which is
		 reliable and fast given the small interval.

7) Step D — refine θ and X via coordinate descent
	 - Perform a small coordinate-descent loop: probe small deltas in θ and X
		 (for each probe recompute the frame and refit M). Accept moves that
		 reduce mse. Reduce step sizes when no improvement is found.
	 - This approach is simple, robust to local curvature, and enforces the
		 assignment bounds at each step.

8) Step E — evaluation / metrics
	 - Sample the fitted parametric curve densely on t in (6,60). For each
		 sampled curve point compute the L1 distance (|dx|+|dy|) to the nearest
		 data point; report mean, median, and tail quantiles (p90, p99, max).
	 - Also produce a Desmos/LaTeX-ready string for submission and save a plot
		 `visualization.png` overlaying the fitted curve and the data.

---

## How to run the python script

1. Make sure all files are present in the folder
2. Create a virtual environment and install dependencies.

```bash
# macOS (zsh)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

3. Run the fitter:

```bash
python3 fit_curve.py
```

The script will print the fitted parameters and save a plot `visualization.png` in the working directory. Example printed output format (the script prints values with numeric precision):

```
=== Fitted Parameters ===
theta (rad)  : 0.123456789
theta (deg)  : 7.070000
M            : 0.001234567
X            : 11.579333333
refit MSE    : 1.234567e-04

Desmos/LaTeX:
\left(t*\cos(0.123457)-e^{0.001235\left|t\right|}\cdot\sin(0.3t)\sin(0.123457)+11.579333,\ 42+t*\sin(0.123457)+e^{0.001235\left|t\right|}\cdot\sin(0.3t)\cos(0.123457)\right)

Saved plot -> visualization.png
```

---

## What the code does?

- Fits a centerline y ≈ s*x + c to the data; derives an initial \(\theta\) via \(\theta=\arctan(s)\) and an initial offset \(X\).
- Rotates data into an along-line (t) and perpendicular (d) frame.
- Solves for the scalar amplitude model `d(t) ≈ exp(M|t|)*sin(0.3 t)` by 1D search for M.
- Uses a coordinate-descent loop to refine (\(\theta\), X) and re-fit M at each step.
- Computes a set-to-set L1 distance from sampled curve points to the nearest data points (reported in the run).

These steps are implemented in `fit_curve.py` & these steps are in-detailed mentioned in the "Detailed Step-by-Step explanation" section in this readme file.
---

## Fitted results (run with provided `XY_DATA.csv`)

I ran `fit_curve.py` on the attached `XY_DATA.csv` and obtained the final fitted parameters below. You can copy the Desmos/LaTeX string into Desmos or include the numeric form in your submission.

- theta (radians): 0.523623214
- theta (degrees): 30.001400
- M: 0.030001685
- X: 55.001164694
- refit MSE: 7.650453719e-07

Desmos/LaTeX string (final output):

\\left(t*\\cos(0.523623)-e^{0.030002\\left|t\\right|}\\cdot\\sin(0.3t)\\sin(0.523623)+55.001165,\\ 42+t*\\sin(0.523623)+e^{0.030002\\left|t\\right|}\\cdot\\sin(0.3t)\\cos(0.523623)\\right)


## Reproducing the L1 score used for assessment

`fit_curve.py` samples the fitted curve at 800 points (t in (6,60)) and computes the mean L1 distance from each sampled curve point to the nearest data point. After running, the script prints a small table of L1 metrics: mean, median, p90, p99, max.

If you prefer a different distance metric (for example, a symmetric set distance or the Euclidean norm), edit the `l1_set_distance` function in `fit_curve.py`.

---


## Notes, assumptions, and tips

- The script enforces the bounds from the assignment when searching (theta in (0, 50 deg), M in (-0.05, 0.05), X in (0,100)).
- If `XY_DATA.csv` is missing, the script will raise `FileNotFoundError`.
- The current algorithm uses brute-force nearest-neighbor L1 distance for the sample sizes in this task. If you scale-up, consider using KD-trees (scipy.spatial.cKDTree) and Euclidean distance for speed.
- For reproducibility, commit `XY_DATA.csv` (if allowed), `fit_curve.py`, `README.md`, and `requirements.txt` to your repo.

---
