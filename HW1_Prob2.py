import numpy as np
import matplotlib.pyplot as plt

#(1)

def int_mid(f, x_min, x_max, N):
    h = np.float32((x_max-x_min)/N)
    bin_N= x_min + h*(np.arange(N, dtype=np.float32)+0.5)
    return np.sum(f(bin_N))*h

def int_trapezoid(f, x_min, x_max, N):
    h = np.float32((x_max-x_min)/N)
    bin_N= x_min + h*(np.arange(N, dtype=np.float32))
    return (np.sum(f(bin_N))-1/2*f(x_min)+1/2*f(x_max))*h

def int_simpsons(f, x_min, x_max, N):
    h = np.float32((x_max-x_min)/N)
    bin_N_odd= x_min + h + 2*h*(np.arange(N/2, dtype=np.float32))
    bin_N_even= x_min + 2*h + 2*h*(np.arange(N/2-1, dtype=np.float32))
    return 1/3*h*(f(x_min)+f(x_max)+4*np.sum(f(bin_N_odd))+2*np.sum(f(bin_N_even)))

#(2)
def plot_error(func_name, func, true_int, x_min, x_max):
    true_value = true_int(x_max)-true_int(x_min)
    N_values = (2**np.arange(1, 20)).astype(int)
    rel_errors_mid, rel_errors_trapezoid, rel_errors_simpons = [], [], []

    for n in N_values:
        fwd_approx = int_mid(func, x_min, x_max, n)
        cen_approx = int_trapezoid(func, x_min, x_max, n)
        ext_approx = int_simpsons(func, x_min, x_max, n)

        rel_errors_mid.append(np.abs((fwd_approx - true_value) / true_value))
        rel_errors_trapezoid.append(np.abs((cen_approx - true_value) / true_value))
        rel_errors_simpons.append(np.abs((ext_approx - true_value) / true_value))

    plt.figure(figsize=(12, 8))
    
    plt.loglog(N_values, rel_errors_mid, 'o-', label='Midpoint Rule', markersize=4)
    plt.loglog(N_values, rel_errors_trapezoid, 's-', label='Trapezoid Rule', markersize=4)
    plt.loglog(N_values, rel_errors_simpons, '^-', label='Simpsons Rule', markersize=4)
    
    plt.title(f"Integration Error for f(x)={func_name} in ({x_min},{x_max})", fontsize=16)
    plt.xlabel("Number of Bins N", fontsize=12)
    plt.ylabel("Relative Error ε", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig("expint.pdf", dpi=300, bbox_inches="tight")
    plt.close()


plot_error('exp(x)', np.exp, np.exp, 0, 1)

