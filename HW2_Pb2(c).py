import math

def multi_relax(a, b, x0=0.5, y0=0.5, tol=1e-8, max_iter=2000):
    x, y = float(x0), float(y0)
    for it in range(1, max_iter+1):
        x_new = math.sqrt(b / y - a)
        y_new = x_new / (a + x_new*x_new)

        if max(abs(x_new - x), abs(y_new - y)) < tol:
            return x_new, y_new, it, True
        x, y = x_new, y_new
    return x, y, max_iter, False

a, b = 1.0, 2.0

x, y, it, ok = multi_relax(a, b)
print(f"converged={ok}, iters={it}, solution (x,y)=({x:.6f},{y:.6f})")
