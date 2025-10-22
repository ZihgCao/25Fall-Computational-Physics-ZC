import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("dow.txt")
t = np.arange(len(y))

plt.figure()
plt.plot(t, y, label="Original")
plt.xlabel("Business day index")
plt.ylabel("DJIA")
plt.title("Dow Jones Industrial Average (raw)")
plt.tight_layout()
plt.show()

def lowpass_irfft(signal, keep_frac):
    ck = np.fft.rfft(signal)
    M = len(ck)
    k_keep = max(1, int(np.ceil(keep_frac * M)))
    ck_f = ck.copy()
    ck_f[k_keep:] = 0
    y_smooth = np.fft.irfft(ck_f, n=len(signal))
    return y_smooth

y_10 = lowpass_irfft(y, keep_frac=0.10)
ck = np.fft.rfft(y)
k = np.arange(len(ck))

# Plot
plt.figure()
plt.plot(k[1:50], ck[1:50])
plt.xlabel("k (Fourier modes)")
plt.ylabel("c_k")
plt.title("Fourier coefficients")
plt.tight_layout()
plt.show()

# Keep 10%
plt.figure()
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, y_10, label="Keep 10% (low-pass)")
plt.xlabel("Business day index")
plt.ylabel("DJIA close")
plt.title("Fourier smoothing: keep first 10% of modes")
plt.legend()
plt.tight_layout()
plt.show()

# Keep 2%
y_02 = lowpass_irfft(y, keep_frac=0.02)

plt.figure()
plt.plot(t, y, label="Original", alpha=0.5)
plt.plot(t, y_02, label="Keep 2% (strong low-pass)")
plt.xlabel("Business day index")
plt.ylabel("DJIA close")
plt.title("Fourier smoothing: keep first 2% of modes")
plt.legend()
plt.tight_layout()
plt.show()