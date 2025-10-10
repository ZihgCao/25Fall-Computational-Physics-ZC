import math

def relaxation_iters(c=2.0, x0=0.5, tol=1e-6, max=10000):
    f = lambda x: 1 - math.exp(-c*x)
    x = float(x0)
    for n in range(1, max+1):
        x_next = f(x)
        if abs(x_next - x) < tol:
            return x_next, n
        x = x_next
    return x, max

root, iters = relaxation_iters()
print(f"x ≈ {root:.6f}, iterations = {iters}")