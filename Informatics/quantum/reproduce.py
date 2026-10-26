"""Independent continuous reference. Python 3 + numpy + scipy.

Install: python -m pip install numpy scipy
Run: python reproduce.py quantum-run-....json
No cloud access, credentials or hardware submissions.
"""
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp

spec = dict(distance_um=5.7, duration_us=4., omega_rad_us=6.3, detuning_rad_us=31.5)
if len(sys.argv) > 1:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    if data.get("model_version") != "rydberg-pair-v1":
        raise ValueError("Unsupported model version")
    spec = data.get("spec", data.get("parameters"))
bounds = {"distance_um": (4.5, 9), "duration_us": (2, 6), "omega_rad_us": (1, 10), "detuning_rad_us": (10, 50)}
for key, (lo, hi) in bounds.items():
    x = spec[key]
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not np.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f"Invalid {key}")
T, omega, detuning = spec["duration_us"], spec["omega_rad_us"], spec["detuning_rad_us"]
V = 5.42e6 / spec["distance_um"] ** 6
X = np.array([[0, 1], [1, 0]])
n = np.diag([0, 1])
I = np.eye(2)
drive = np.kron(X, I) + np.kron(I, X)
onsite = np.kron(n, I) + np.kron(I, n)
interaction = np.kron(n, n)

def rhs(t, psi):
    f = t / T
    amplitude = omega * max(0., min(1., f / .025, (1 - f) / .025))
    delta = np.interp(t, [0, .025 * T, .975 * T, T], [-detuning, -detuning, detuning, detuning])
    H = amplitude / 2 * drive - delta * onsite + V * interaction
    return -1j * H @ psi

state = np.array([1, 0, 0, 0], dtype=complex)
for a, b in zip([0, .025 * T, .975 * T], [.025 * T, .975 * T, T]):
    result = solve_ivp(rhs, (a, b), state, method="DOP853", rtol=1e-11, atol=1e-13)
    if not result.success:
        raise RuntimeError(result.message)
    state = result.y[:, -1]
probabilities = dict(zip(["gg", "gr", "rg", "rr"], np.abs(state) ** 2))
print(json.dumps({"model_version": "rydberg-pair-v1", "spec": spec, "probabilities": probabilities,
                  "norm_error": float(abs(sum(probabilities.values()) - 1))}, indent=2))
