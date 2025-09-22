import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

k, Pk = np.loadtxt("lcdm_z0.matter_pk", usecols=(0, 1), unpack=True)
spline = CubicSpline(k, Pk)

def P_itpl(kk):
    return spline(kk)

def xi(r, kmin, kmax, npts=200000):
    k_val = np.linspace(kmin, kmax, npts)
    Pk_val = spline(k_val)
    integrand = k_val**2 * Pk_val * np.sin(k_val*r)/(k_val*r)
    return np.trapz(integrand, k_val) / (2*np.pi**2)

r_val = np.linspace(50, 120, 200)
xi_val = [xi(r, k.min(), k.max()) for r in r_val]
r2xi_val = [r**2*xi(r, k.min(), k.max()) for r in r_val]

# Plot
plt.figure(figsize=(10,6))
plt.plot(r_val, r2xi_val, '-')
plt.title(r"$r^2 \xi(r)$ with Trapezoidal Integration", fontsize=16)
plt.xlabel("Separation, r [Mpc/h]", fontsize=12)
plt.ylabel(r"$r^2 \xi(r)$", fontsize=12)
plt.grid(True)
plt.legend()

plt.savefig("correlator.pdf", dpi=300, bbox_inches="tight")
plt.close()