import numpy as np
import matplotlib.pyplot as plt

#(1)

def forward_difference(f, x, h):
    return (np.float32(f(x + h)) - np.float32(f(x))) / np.float32(h)

def central_difference(f, x, h):
    return (np.float32(f(x + h)) - np.float32(f(x - h))) / (2 * np.float32(h))

def extrapolated_difference(f, x, h):
    return (-np.float32(f(x + 2*h))+ 8 * np.float32(f(x + h))-8 * np.float32(f(x - h))+np.float32(f(x - 2*h)))/(12*np.float32(h))

#(2)
def plot_error(func_name, func, true_deriv, x_val, filename):
    x = np.float32(x_val)
    true_value = true_deriv(x_val)
    h_values = np.logspace(-8, 0, 100, dtype=np.float32)
    rel_errors_fwd, rel_errors_cen, rel_errors_ext = [], [], []

    for h in h_values:
        fwd_approx = forward_difference(func, x, h)
        cen_approx = central_difference(func, x, h)
        ext_approx = extrapolated_difference(func, x, h)

        rel_errors_fwd.append(np.abs((fwd_approx - true_value) / true_value))
        rel_errors_cen.append(np.abs((cen_approx - true_value) / true_value))
        rel_errors_ext.append(np.abs((ext_approx - true_value) / true_value))

    plt.figure(figsize=(12, 8))
    plt.loglog(h_values, rel_errors_fwd, 'o-', label='Forward', markersize=4)
    plt.loglog(h_values, rel_errors_cen, 's-', label='Central', markersize=4)
    plt.loglog(h_values, rel_errors_ext, '^-', label='Extrapolated', markersize=4)

    plt.title(f"Differentiation Error for f(x)={func_name} at x={x_val}", fontsize=16)
    plt.xlabel("Step Size h", fontsize=12)
    plt.ylabel("Relative Error ε", fontsize=12)
    plt.legend()
    plt.ylim(1e-7, 1e1)
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="--")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

plot_error("cos(x)", np.cos, lambda x: -np.sin(x), 0.1, "cos_x0.1.pdf")
plot_error("cos(x)", np.cos, lambda x: -np.sin(x), 10, "cos_x10.pdf")
plot_error("exp(x)", np.exp, lambda x: np.exp(x), 0.1, "exp_x0.1.pdf")
plot_error("exp(x)", np.exp, lambda x: np.exp(x), 10, "exp_x10.pdf")


