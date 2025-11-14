import numpy as np
import matplotlib.pyplot as plt

a = 1.0
b = 3.0

def f(r, t):
    x = r[0]
    y = r[1]
    f_x = 1 - (b + 1) * x + a * x**2 * y
    f_y = b * x - a * x**2 * y
    
    return np.array([f_x, f_y], float)

# midpoint method

def modified_midpoint_step(r, t, H, n):
    h = H / n
    
    r_current = np.copy(r)
    r_next = r_current + h * f(r_current, t)


    for i in range(1, n):
        t_current = t + i * h
        r_temp = r_next 
        r_next = r_current + 2 * h * f(r_next, t_current)
        r_current = r_temp

    t_final = t + H
    r_final = 0.5 * (r_current + r_next + h * f(r_next, t_final))
    
    return r_final

# Bulirsch-Stoer method, following 8.5.5, using the above defined function

def bulirsch_step(r, t, H):
    n_sequence = [2, 4, 6, 8, 10, 12, 14, 16]
    num_rows = len(n_sequence)
    
    R = [[np.zeros(2, float) for _ in range(i + 1)] for i in range(num_rows)]


    for i in range(num_rows):
        n = n_sequence[i]
        R[i][0] = modified_midpoint_step(r, t, H, n)

    for m in range(1, num_rows):
        for i in range(m, num_rows):

            n_i = n_sequence[i]
            n_i_minus_m = n_sequence[i - m]
            
            ratio_sq = (n_i / n_i_minus_m)**2
            
            numerator = R[i][m - 1] - R[i - 1][m - 1]
            denominator = ratio_sq - 1.0

            R[i][m] = R[i][m - 1] + (numerator / denominator)

    # final result from extrapolation
    r_final = R[num_rows - 1][num_rows - 1]
    error_vector = R[num_rows - 1][num_rows - 1] - R[num_rows - 1][num_rows - 2]
    
    return r_final, error_vector

# Recursive, using the above defined function
t_points = []
x_points = []
y_points = []

DELTA = 1e-10

def solve_recursive(r_start, t_start, H):

    r_final_attempt, error_vector = bulirsch_step(r_start, t_start, H)

    error_x = abs(error_vector[0])
    error_y = abs(error_vector[1])
    

    target_error = H * DELTA 
    

    if error_x < target_error and error_y < target_error:
        # :-)
        t_points.append(t_start + H)
        x_points.append(r_final_attempt[0])
        y_points.append(r_final_attempt[1])

        return r_final_attempt
        

    else:
        # :-(
        r_midpoint = solve_recursive(r_start, t_start, H / 2.0)
        r_final = solve_recursive(r_midpoint, t_start + H / 2.0, H / 2.0)

        return r_final




def solve_brusselator():

    t_points.clear()
    x_points.clear()
    y_points.clear()

    t_start = 0.0
    t_end = 20.0
    r_initial = np.array([0.0, 0.0], float)

    t_points.append(t_start)
    x_points.append(r_initial[0])
    y_points.append(r_initial[1])

    
    solve_recursive(r_initial, t_start, t_end)



    plt.figure(figsize=(12, 7))

    plt.plot(t_points, x_points, 'b-', label='x(t)', alpha=0.7)
    plt.plot(t_points, y_points, 'g-', label='y(t)', alpha=0.7)

    plt.plot(t_points, x_points, 'bo', markersize=4, label='x(t) steps')
    plt.plot(t_points, y_points, 'go', markersize=4, label='y(t) steps')
    
    plt.xlabel("Time (t)")
    plt.ylabel("Concentration")
    plt.title("Brusselator Chemical Oscillator (Adaptive Bulirsch-Stoer)")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


solve_brusselator()