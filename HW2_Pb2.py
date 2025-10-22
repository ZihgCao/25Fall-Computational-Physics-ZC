import math

def f(x):
    return 5*math.exp(-x) + x - 5

a, b = 1, 10.0      # since we can see that f(1)<0, f(10)>0
eps = 1e-6
while b - a > eps:
    m = 0.5*(a + b)
    if f(a)*f(m) <= 0:
        b = m
    else:
        a = m
x = 0.5*(a + b)

h  = 6.62607015e-34        # (J·s)
c  = 2.99792458e8          # (m/s)
kB = 1.380649e-23          # (J/K)
wien_b = (h*c)/(kB*x)      # (m·K)


lambda_peak = 502e-9       # (m)
T_sun = wien_b / lambda_peak

print(f"x (root)       = {x:.9f}")
print(f"Wien constant b = {wien_b:.9e} m·K")
print(f"Sun temperature ≈ {T_sun:.1f} K (for λ_peak = 502 nm)")