import math
import numpy as np
import matplotlib.pyplot as plt

def overrelaxation(c=2.0, omega=0.5, x0=0.5, tol=1e-6, max=5000):
    f = lambda x: 1.0 - math.exp(-c * x)
    x = float(x0)
    for n in range(1, max + 1):
        x_new = (1 + omega) * f(x) - omega * x
        if abs(x_new - x) < tol:
            return x_new, n, True
        x = x_new
    return x, max, False


c = 2.0
x0 = 0.5
tol = 1e-6

omegas = np.linspace(0.0, 1.2, 121)
iters = []
for w in omegas:
    _, it, ok = overrelaxation(c=c, omega=w, x0=x0, tol=tol)
    iters.append(it if ok else np.nan)

# Plot
plt.plot(omegas, iters, lw=1.8)
plt.xlabel("ω (overrelaxation parameter)")
plt.ylabel("iterations to reach |Δx| < 10⁻⁶")
plt.title("Overrelaxation for  x = 1 - e^{-2x}")
plt.grid(True)
plt.show()
