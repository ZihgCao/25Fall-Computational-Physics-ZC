import numpy as np
import matplotlib.pyplot as plt

t, y = np.loadtxt("sunspots.txt", unpack=True)

plt.figure()
plt.plot(t, y)
plt.xlabel("Month since Jan 1749")
plt.ylabel("Sunspots")
plt.title("Sunspots vs Time")
plt.tight_layout()
plt.show()

ck = np.fft.rfft(y)
power = np.abs(ck)**2
k = np.arange(len(power))

# Plot
max_k = 50
plt.figure()
plt.plot(k[1:max_k], power[1:max_k])
plt.xlabel("k (Fourier modes)")
plt.ylabel("|c_k|^2 (power)")
plt.title("Power Spectrum of Sunspots")
plt.tight_layout()
plt.show()

N = len(y)
k_peak = 1 + np.argmax(power[1:])
period_months = N / k_peak
period_years = period_months / 12.0

print(f"k_peak = {k_peak}")
print(f"Estimated period ≈ {period_months:.2f} months (~{period_years:.2f} years)")