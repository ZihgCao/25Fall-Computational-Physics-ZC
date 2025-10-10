import math

def multi_relax(a=1.0, b=2.0, x0=0.1, y0=0.1, tol=1e-8, max_iter=2000):
    x, y = float(x0), float(y0)
    for it in range(1, max_iter+1):
        x_new = y * (a + x*x)
        y_new = b / (a + x*x)
        if max(abs(x_new - x), abs(y_new - y)) < tol:
            return x_new, y_new, it, True
        x, y = x_new, y_new
    return x, y, max_iter, False

_, _, _, ok = multi_relax()
print(f"{ok}")