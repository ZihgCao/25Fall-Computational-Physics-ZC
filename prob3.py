import numpy as np
import matplotlib.pyplot as plt

blur = np.loadtxt("blur.txt")
H, W = blur.shape

def gaussian_psf(shape, sigma):
    H, W = shape
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    dy = np.minimum(y, H - y)
    dx = np.minimum(x, W - x)
    g = np.exp(-(dx**2 + dy**2) / (2*sigma**2))
    g /= g.sum()
    return g

sigma = 25.0
psf = gaussian_psf((H, W), sigma)

plt.figure(figsize=(4.5, 4))
plt.imshow(blur, cmap="gray", origin="upper") 
plt.title("Blurred image")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4.5, 4))
plt.imshow(psf, cmap="gray", origin="upper") 
plt.title(f"PSF (σ={sigma:g})")
plt.axis("off")
plt.tight_layout()
plt.show()


Bhat = np.fft.rfft2(blur)
Fhat = np.fft.rfft2(psf)
eps = 1e-3
mask = np.abs(Fhat) > eps
Ahat = np.zeros_like(Bhat)
Ahat[mask] = Bhat[mask] / Fhat[mask]
deblur = np.fft.irfft2(Ahat, s=blur.shape)

plt.figure(figsize=(9.2, 4))
plt.subplot(1, 2, 1)
plt.imshow(blur, cmap="gray", origin="upper")   # ← fixed
plt.title("Input (blurred)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(deblur, cmap="gray", origin="upper") # ← fixed
plt.title("Output (deblurred)")
plt.axis("off")
plt.tight_layout()
plt.show()