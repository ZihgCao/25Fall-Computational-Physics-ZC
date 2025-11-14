import numpy as np
import matplotlib.pyplot as plt
import time


def derivatives(y, t, A, B):

    # EOM
    x, y, vx, vy = y
    
    r_sq = x*x + y*y
    r = np.sqrt(r_sq)
    
    if r == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])

    r3 = r * r_sq
    
    v_sq = vx*vx + vy*vy
    v = np.sqrt(v_sq)
    

    g_factor = -1.0 / (4.0 * r3)
    ax_g = g_factor * x
    ay_g = g_factor * y

    if A == 0.0 or v == 0.0:
        ax_df = 0.0
        ay_df = 0.0
    else:
        v3 = v * v_sq
        df_factor = -A / (v3 + B)
        ax_df = df_factor * vx
        ay_df = df_factor * vy
        
    return np.array([vx, vy, ax_g + ax_df, ay_g + ay_df])

def get_energy(y):
    x, y, vx, vy = y
    r = np.sqrt(x*x + y*y)
    v_sq = vx*vx + vy*vy
    if r == 0:
        return np.inf
    return 0.5 * v_sq - 1.0 / (4.0 * r)

def get_ang_mom(y):
    # Angular Momentum
    x, y, vx, vy = y
    return x * vy - y * vx



def rk4_step(func, y, t, dt, *args):
    # RK core
    k1 = func(y, t, *args)
    k2 = func(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = func(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = func(y + dt * k3, t + dt, *args)
    
    y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y_new

def adaptive_rk4_step(func, y, t, dt, tol_per_time, *args):

    # Adaptive RK
    safety = 0.9
    min_scale = 0.2
    max_scale = 5.0

    while True:

        y1 = rk4_step(func, y, t, dt, *args)
        

        y2_half = rk4_step(func, y, t, dt/2.0, *args)
        y2 = rk4_step(func, y2_half, t + dt/2.0, dt/2.0, *args)
        

        err_x = np.abs(y2[0] - y1[0]) / 15.0
        err_y = np.abs(y2[1] - y1[1]) / 15.0
        err_estimate_abs = np.sqrt(err_x**2 + err_y**2)
        
        target_err_abs = tol_per_time * dt
        
        if err_estimate_abs == 0:
            dt_scale = max_scale
        else:
            dt_scale = safety * (target_err_abs / err_estimate_abs)**0.2
        
        dt_scale = min(max_scale, max(min_scale, dt_scale))
        
        if err_estimate_abs < target_err_abs:
            t_new = t + dt
            y_new = y2
            dt_new = dt * dt_scale
            return y_new, t_new, dt_new
        else:
            dt = dt * dt_scale
            if dt < 1e-15:
                raise

def integrate(func, y0, t0, t_end, dt_initial, tol_per_time, *args, stop_r=None):

    y = y0
    t = t0
    dt = dt_initial

    history_t = [t0]
    history_y = [y0]
    
    while t < t_end:

        if t + dt > t_end:
            dt = t_end - t
            
        if dt <= 0:
            break

        y_new, t_new, dt_new = adaptive_rk4_step(func, y, t, dt, tol_per_time, *args)
        
        y = y_new
        t = t_new
        dt = dt_new
        
        history_t.append(t)
        history_y.append(y)
        
        if stop_r is not None:
            r = np.sqrt(y[0]**2 + y[1]**2)
            if r < stop_r:
                break
      
    return np.array(history_t), np.array(history_y)



def run_part_a(tol_per_time):

    v0_a = np.sqrt(5e-8)
    y0_a = np.array([1.0, 0.0, 0.0, v0_a])

    period = np.pi * np.sqrt(2.0)
    t_end_a = 10 * period # 10 orbits
    
    t, y = integrate(derivatives, y0_a, 0.0, t_end_a, 0.01, 
                     tol_per_time, 0.0, 0.0, stop_r=None)
    

    energies = np.array([get_energy(yi) for yi in y])
    ang_moms = np.array([get_ang_mom(yi) for yi in y])
    
    e0 = energies[0]
    l0 = ang_moms[0]
    
    rel_err_E = np.abs((energies - e0) / e0)
    rel_err_L = np.abs((ang_moms - l0) / l0)
    

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(t, rel_err_E, label='Relative Energy Error')
    ax1.set_ylabel('| (E(t) - E0) / E0 |')
    ax1.set_yscale('log')

    ax1.set_title(f'Conservation Errors (tol = {tol_per_time:.1e})')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t, rel_err_L, label='Relative Ang. Mom. Error', color='orange')
    ax2.set_ylabel('| (L(t) - L0) / L0 |')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('part_a_conservation.png')
    plt.show()


def run_part_b(tol_per_time):
    # (b)
    y0_b = np.array([1.0, 0.0, 0.0, 0.4])
    A, B = 1.0, 1.0
    

    t_end_b = 1e8
    rs = 1e-7 # Schwarzschild
    
    t, y = integrate(derivatives, y0_b, 0.0, t_end_b, 0.01,
                     tol_per_time, A, B, stop_r=rs)
                     
    x = y[:, 0]
    y_pos = y[:, 1]
    r = np.sqrt(x**2 + y_pos**2)

    plt.figure(figsize=(8, 8))
    plt.plot(x, y_pos)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title('BH Orbital Path (A=1, B=1)')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('part_b_path.png')
    plt.show()
    

    plt.figure(figsize=(10, 6))
    plt.plot(t, np.log10(r))
    plt.xlabel('Time')
    plt.ylabel('log10(r)')

    plt.title('Radius Decay (A=1, B=1)')
    plt.grid(True)
    plt.savefig('part_b_radius_vs_time.png')
    plt.show()
    
    return t, y

def run_part_c(tol_per_time):
    # (c)

    
    y0_c = np.array([1.0, 0.0, 0.0, 0.4])
    rs = 1e-7
    t_end_c = 1e8
    
    # pairs, for simplicity
    param_pairs = [
        (0.5, 2.0),
        (0.6, 2.0),
        (0.75, 2.0),
        (0.75, 1.75),
        (0.75, 1.5),
        (1.0, 1.0),
        (1.5, 0.75),
        (2.0, 0.5),
        (4.0, 0.5),
        (6.0, 0.5),
        (8.0, 0.5),
        (10.0, 0.5)
    ]
    
    B_over_A_ratios = [B / A for A, B in param_pairs]
    
    times_to_reach_rs = []
    

    
    for A, B in param_pairs:
        t, _ = integrate(derivatives, y0_c, 0.0, t_end_c, 0.01,
                         tol_per_time, A, B, stop_r=rs)
        times_to_reach_rs.append(t[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(B_over_A_ratios, times_to_reach_rs, 'bo-')
    plt.xlabel('Ratio B/A')
    plt.ylabel(f'Time to Reach r_s')

    plt.title('Decay Time vs. B/A Ratio')
    plt.grid(True)
    plt.savefig('part_c_time_vs_ratio.png')
    plt.show()
   

def run_part_d(tol_per_time):
    #(d)

    rs = 1e-7
    t_end_d = 1e8
    
    v0_list = [0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8]
    times_v0 = []
    
    for v0 in v0_list:
        y0_d = np.array([1.0, 0.0, 0.0, v0])
        t, _ = integrate(derivatives, y0_d, 0.0, t_end_d, 0.01,
                         tol_per_time, 1.0, 1.0, stop_r=rs)
        times_v0.append(t[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(v0_list, times_v0, 'go-')
    plt.xlabel('Initial Velocity (v_y at x=1)')
    plt.ylabel(f'Time to Reach r_s')

    plt.title('Decay Time vs. Initial Velocity (A=1, B=1)')
    plt.grid(True)
    plt.savefig('part_d_time_vs_v0.png')
    plt.show()

    param_sets = {
        'B/A = 1': [
            {'A': 0.5, 'B': 0.5, 'label': 'A=0.5, B=0.5'},
            {'A': 1.0, 'B': 1.0, 'label': 'A=1.0, B=1.0'},
            {'A': 2.0, 'B': 2.0, 'label': 'A=2.0, B=2.0'},
            {'A': 4.0, 'B': 4.0, 'label': 'A=4.0, B=4.0'},
            {'A': 6.0, 'B': 6.0, 'label': 'A=6.0, B=6.0'},
        ],
        'B/A = 2': [
            {'A': 0.5, 'B': 1.0, 'label': 'A=0.5, B=1.0'},
            {'A': 1.0, 'B': 2.0, 'label': 'A=1.0, B=2.0'},
            {'A': 2.0, 'B': 4.0, 'label': 'A=2.0, B=4.0'},
            {'A': 4.0, 'B': 8.0, 'label': 'A=4.0, B=8.0'},
            {'A': 6.0, 'B': 12.0, 'label': 'A=6.0, B=12.0'},
        ]
    }
    
    results = {}
    y0_d = np.array([1.0, 0.0, 0.0, 0.4])

    for ratio_label, param_list in param_sets.items():
        results[ratio_label] = {'A_vals': [], 't_vals': []}
        for params in param_list:
            t, y = integrate(derivatives, y0_d, 0.0, t_end_d, 0.01,
                             tol_per_time, params['A'], params['B'], stop_r=rs)
            
            t_reach = t[-1]
            results[ratio_label]['A_vals'].append(params['A'])
            results[ratio_label]['t_vals'].append(t_reach)



    plt.figure(figsize=(10, 6))
    
    # Plot for B/A = 1
    data_ba1 = results['B/A = 1']
    plt.plot(data_ba1['A_vals'], data_ba1['t_vals'], 'bo-', label='B/A = 1')
    
    # Plot for B/A = 2
    data_ba2 = results['B/A = 2']
    plt.plot(data_ba2['A_vals'], data_ba2['t_vals'], 'ro-', label='B/A = 2')

    plt.xlabel('Parameter A')
    plt.ylabel(f'Time to Reach r_s')
    plt.title('Decay Time vs. A (for fixed B/A ratios)')
    plt.legend()
    plt.grid(True)
    plt.savefig('part_d_time_vs_A.png')
    plt.show()



TOLERANCE_D = 1e-7


run_part_a(TOLERANCE_D)
run_part_b(TOLERANCE_D)
run_part_c(TOLERANCE_D)
run_part_d(TOLERANCE_D)
