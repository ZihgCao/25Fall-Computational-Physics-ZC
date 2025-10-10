import numpy as np
import matplotlib.pyplot as plt

def numgrad(f, params, h=1e-4):
    params = list(map(float, params))
    n = len(params)
    g = np.zeros(n)
    for i in range(n):
        p_plus  = params.copy(); p_plus[i]  += h
        p_minus = params.copy(); p_minus[i] -= h
        g[i] = (f(*p_plus) - f(*p_minus)) / (2.0*h)
    return g

#Here we use a modified grad_descent method.
def grad_descent(f, params0, lr=1e-2, max_iter=20000, tol=1e-6,h=1e-4):
    p = np.array(params0, float)
    hist = []
    fx = f(*p)
    for _ in range(max_iter):
        hist.append(fx)
        g = numgrad(f, p, h)
        if np.max(np.abs(g)) < tol: break
        step = (lr if np.isscalar(lr) else np.array(lr))*g
        t = 1.0
        while True:
            p_new = p - t*step
            fx_new = f(*p_new)
            if np.isfinite(fx_new) and fx_new <= fx: break
            t *= 0.5
            if t < 1e-8: break
        p, fx = p_new, fx_new
    return p, np.array(hist)


def f_test(x, y): return (x-2)**2 + (y-2)**2
xy_star, f_hist = grad_descent(f_test, params0=[-3, 5], lr=[0.1, 0.1])
print("[test] min ~", xy_star, " f_min=", f_hist[-1])
plt.figure(); plt.plot(f_hist); plt.xlabel("iter"); plt.ylabel("f(x,y)"); plt.title("Test GD"); plt.grid(True)


logM, n_obs, n_err = np.loadtxt("smf_cosmos.dat").T
M = 10**logM

def schechter_logp(logphi, logMstar, alpha,  Mgal):
    phi, Mstar = 10**logphi, 10**logMstar
    return phi * ( Mgal/Mstar)**(alpha+1) * np.exp(- Mgal/Mstar) * np.log(10)

def chi2(logphi, logMstar, alpha):
    model = schechter_logp(logphi, logMstar, alpha, M)
    return np.sum(((n_obs - model)/n_err)**2)


p0 = [-3.0, 10.0, -1.0]                 # (log φ*, log M*, alpha)
lr = 1e-2

p_best, chi_hist = grad_descent(chi2, p0, lr=lr, h=1e-6)
logphi_b, logMstar_b, alpha_b = p_best
print("[fit] log phi*, logM*, alpha =", p_best, "  chi2_min=", chi_hist[-1])


plt.figure(); plt.plot(chi_hist)
plt.xlabel("iteration"); plt.ylabel(r"$\chi^2$"); plt.title(r"$\chi^2$ vs iteration"); plt.grid(True)


Mf = np.logspace(logM.min()-0.5, logM.max()+0.5, 300)
nf = schechter_logp(logphi_b, logMstar_b, alpha_b, Mf)
plt.figure()
plt.errorbar(M, n_obs, yerr=n_err, fmt='o', ms=4, label="data")
plt.plot(Mf, nf, lw=2, label="best-fit")
plt.xscale('log'); plt.yscale('log'); plt.legend()
plt.xlabel(r"$M_{\rm gal}$"); plt.ylabel(r"$n(M_{\rm gal})$")
plt.title("Schechter fit (COSMOS)"); plt.grid(True, which='both', ls='--', alpha=0.6)
plt.show()