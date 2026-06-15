import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.optimize import minimize
import csv

# ==============================================================================
# AIRFOIL GENERATION FUNCTIONS
# ==============================================================================

def naca4(m_param, p_param, t_param, c=1.0, n=100):
    # Cosine Spacing: densifies points at the leading and trailing edges.
    # Essential to prevent XFOIL from failing and detaching the flow at low Reynolds numbers
    beta = np.linspace(0, np.pi, n)
    x = c * (0.5 * (1 - np.cos(beta)))
    yt = 5 * t_param * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

    if p_param == 0 or m_param == 0:
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        front_x = x[x < p_param * c]
        back_x = x[x >= p_param * c]

        if len(front_x) > 0:
            yc_front = (m_param / p_param**2) * (2 * p_param * (front_x / c) - (front_x / c)**2)
            dyc_dx_front = (2 * m_param / p_param**2) * (p_param - front_x / c)
            yc[:len(front_x)] = yc_front
            dyc_dx[:len(front_x)] = dyc_dx_front

        if len(back_x) > 0:
            yc_back = (m_param / (1 - p_param)**2) * ((1 - 2 * p_param) + 2 * p_param * (back_x / c) - (back_x / c)**2)
            dyc_dx_back = (2 * m_param / (1 - p_param)**2) * (p_param - back_x / c)
            yc[len(front_x):] = yc_back
            dyc_dx[len(front_x):] = dyc_dx_back
            
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

    X = np.concatenate((np.flip(xu), xl[1:]))
    Y = np.concatenate((np.flip(yu), yl[1:]))
    
    return X, Y, (xu, yu, xl, yl)

def save_airfoil_to_file(X, Y, filename):
    with open(filename, "w") as f:
        for i in range(len(X)):
            f.write(f"{X[i]:.6f} {Y[i]:.6f}\n")

# ==============================================================================
# CFD ANALYSIS FUNCTION (XFOIL WRAPPER)
# ==============================================================================

def run_xfoil_analysis(airfoil_file, alpha, Re, Mach=0.0):
    xfoil_input_file = "xfoil_input.in"
    polar_file = "polar.dat"

    # Clean up previous files to prevent XFOIL from freezing on "Overwrite (Y/N)?"
    for f in [xfoil_input_file, polar_file]:
        if os.path.exists(f):
            os.remove(f)

    with open(xfoil_input_file, "w") as f:
        f.write(f"LOAD {airfoil_file}\n")
        f.write("PANE\n")  
        f.write("OPER\n")
        
        # TRICK 1: Inviscid warm-up (stabilizes the boundary layer)
        f.write("ALFA 0\n") 
             
        f.write(f"Visc {Re}\n")
        f.write(f"Mach {Mach}\n")
        f.write("ITER 500\n") 
        
        # Activate PACC *before* the sweep to ensure we record them all
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n") 
        
        if alpha == 0.0:
            f.write("ALFA 0.0\n")
        else:
            step = 1.0 if alpha > 0 else -1.0
            # Sweep ramp using ASEQ for robustness
            f.write(f"ASEQ 0.0 {alpha - step/2} {step}\n")
            # Force the exact last angle
            f.write(f"ALFA {alpha}\n")
            
        f.write("\n")      
 
        f.write("QUIT\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")

    try:
        with open(xfoil_input_file, "r") as stdin_file:
            subprocess.run([xfoil_exe_path], stdin=stdin_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # If it times out or fails, we still proceed to read the recovered partial data!
        pass

    cl, cd, achieved_alpha = None, None, None
    try:
        with open(polar_file, "r") as f:
            lines = [line for line in f if not line.startswith("#") and len(line.strip()) > 0]
            best_diff = 1e6
            for line in lines:
                data = line.split()
                if len(data) >= 3:
                    try:
                        a_val = float(data[0])
                        cl_val = float(data[1])
                        cd_val = float(data[2])
                        
                        diff = abs(a_val - alpha)
                        if diff < best_diff:
                            best_diff = diff
                            achieved_alpha = a_val
                            cl = cl_val
                            cd = cd_val
                    except ValueError:
                        pass
    except (IOError, IndexError):
        pass
    
    for f in [xfoil_input_file, polar_file, airfoil_file]:
        if os.path.exists(f):
            os.remove(f)
            
    return cl, cd, achieved_alpha

# ==============================================================================
# REYNOLDS CALCULATION AND OBJECTIVE FUNCTION
# ==============================================================================

def calculate_reynolds(speed, chord, kinematic_viscosity):
    return (speed * chord) / kinematic_viscosity

def objective_function(params, Re, alpha, target_cl, max_cd):
    objective_function.eval_count += 1
    eval_count = objective_function.eval_count
    m, p, t = params
    
    if not (0.0 <= m <= 0.20 and 0.1 <= p <= 0.8 and 0.05 <= t <= 0.35):
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} |  Bounds  |  Bounds  | {1e6:.4e} |")
        objective_function.history.append([eval_count, m, p, t, "Bounds", "Bounds", 1e6])
        return 1e6

    airfoil_name = f"temp_naca.dat"
    X, Y, _ = naca4(m, p, t)
    save_airfoil_to_file(X, Y, airfoil_name)
    
    cl, cd, achieved_alpha = run_xfoil_analysis(airfoil_name, alpha, Re)
    
    if cl is not None and cd is not None and cd > 0:
        # We use squared error instead of abs() to make the function "smooth" (differentiable) for SLSQP
        cl_error = ((cl - target_cl) * 10) ** 2
        cd_penalty = (max(0, cd - max_cd) * 1000) ** 2
        
        # If XFOIL loses adhesion (stalls) BEFORE reaching the requested angle:
        if abs(achieved_alpha - alpha) > 0.1:
            alpha_penalty = (abs(achieved_alpha - alpha) * 1000) ** 2
            total_score = cl_error + cd_penalty + alpha_penalty
            print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} | {cl:.4f}*  | {cd:.5f}* | {total_score:.4e} |")
            objective_function.history.append([eval_count, m, p, t, cl, cd, total_score])
            return total_score
            
        total_score = cl_error + cd_penalty
        
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} |  {cl:.4f}  |  {cd:.5f} | {total_score:.4e} |")
        objective_function.history.append([eval_count, m, p, t, cl, cd, total_score])
        return total_score
    else:
        # Emergency gradient: If it fails even at 0°, create a slope towards thicker and more stable wings
        emergency = 1e6 + ((0.12 - t)**2 * 1e5) + ((0.05 - m)**2 * 1e5)
        print(f"| {eval_count:4d} | {m:.4f} | {p:.4f} | {t:.4f} | Separated | Separated | {emergency:.4e} |")
        objective_function.history.append([eval_count, m, p, t, "Separated", "Separated", emergency])
        return emergency

# ==============================================================================
# INTERACTIVE INPUT FUNCTIONS
# ==============================================================================

def get_float_input(prompt):
    while True:
        val = input(f"{prompt}: ").strip()
        try:
            return float(val)
        except ValueError:
            print("Error: invalid input. A numerical value is required.")

def get_fluid_selection():
    fluids = {
        '1': {'name': 'Air (Standard SL)', 'viscosity': 1.46e-5},
        '2': {'name': 'Water (20 degrees)', 'viscosity': 1.00e-6}
    }
    
    while True:
        print("\nSelect the operating fluid:")
        print("1. Air (Kinematic viscosity: 1.46e-5 m^2/s)")
        print("2. Water (Kinematic viscosity: 1.00e-6 m^2/s)")
        choice = input("Choice (1 or 2): ").strip()
        
        if choice in fluids:
            return fluids[choice]
        print("Error: invalid selection.")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    if not os.path.exists(xfoil_exe_path):
        print("CRITICAL ERROR: 'xfoil.exe' executable not found.")
        print(f"Checked path: {script_dir}")
        exit()

    print("======================================================================")
    print(" AIRFOIL OPTIMIZATION (INVERSE DESIGN)")
    print("======================================================================")
    
    # Acquisition of operating parameters for Reynolds calculation
    fluid = get_fluid_selection()
    speed = get_float_input("Enter the design speed (m/s)")
    chord = get_float_input("Enter the airfoil chord length (m)")
    
    TARGET_REYNOLDS = calculate_reynolds(speed, chord, fluid['viscosity'])
    
    # Acquisition of optimization parameters
    TARGET_ALPHA = get_float_input("\nEnter the angle of attack (degrees)")
    TARGET_CL = get_float_input("Enter the target Lift Coefficient (Cl)")
    MAX_CD = get_float_input("Enter the maximum tolerated Drag Coefficient (Cd)")
    
    # Starting from a "quasi-flat" airfoil (m=0.01) instead of m=0.0 to avoid blocking on "p" gradients
    initial_guess = [0.01, 0.4, 0.12]
    bounds = [(0.0, 0.20), (0.1, 0.8), (0.05, 0.35)]

    print("\n----------------------------------------------------------------------")
    print(f"Fluid: {fluid['name']} | Speed: {speed} m/s | Chord: {chord} m")
    print(f"Calculated Reynolds Number: {TARGET_REYNOLDS:.1f}")
    print(f"Objective: Cl = {TARGET_CL} | Constraint: Cd <= {MAX_CD} at {TARGET_ALPHA} degrees")
    print("Initial airfoil: NACA 1412 (Quasi-Symmetric)")
    print("----------------------------------------------------------------------")
    
    print(f"+------+--------+--------+--------+----------+----------+--------------+")
    print(f"| Eval |   m    |   p    |   t    |    Cl    |    Cd    |    Error     |")
    print(f"+------+--------+--------+--------+----------+----------+--------------+")

    objective_function.eval_count = 0
    objective_function.history = []

    result = minimize(
        objective_function,
        initial_guess,
        args=(TARGET_REYNOLDS, TARGET_ALPHA, TARGET_CL, MAX_CD),
        method='SLSQP',
        bounds=bounds,
        options={
            'disp': True, 
            'maxiter': 2000,
            'ftol': 1e-4, # Lowered tolerance: accounts for the intrinsic numerical noise of XFOIL
            'eps': 1e-4
        }
    )

    print("+------+--------+--------+--------+----------+----------+--------------+")
    print("\n======================================================================")
    print(" OPTIMIZATION RESULTS")
    print("======================================================================")
    
    if result.success or result.nfev > 0:
        optimal_params = result.x
        
        airfoil_name = "final_naca.dat"
        X, Y, coords_optimal = naca4(optimal_params[0], optimal_params[1], optimal_params[2])
        save_airfoil_to_file(X, Y, airfoil_name)
        final_cl, final_cd, final_alpha = run_xfoil_analysis(airfoil_name, TARGET_ALPHA, TARGET_REYNOLDS)
        if os.path.exists(airfoil_name): os.remove(airfoil_name)
        
        m_opt_str = str(int(round(optimal_params[0] * 100)))
        p_opt_str = str(int(round(optimal_params[1] * 10)))
        t_opt_str = f"{int(round(optimal_params[2] * 100)):02d}"
        naca_opt_str = f"{m_opt_str}{p_opt_str}{t_opt_str}"

        # --- FOLDER CREATION AND DATA EXPORT ---
        folder_name = f"Results_Re{int(TARGET_REYNOLDS)}_Alpha{TARGET_ALPHA}_Cl{TARGET_CL}"
        output_dir = os.path.join(script_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV Export
        csv_path = os.path.join(output_dir, "optimization_history.csv")
        with open(csv_path, mode="w", newline="") as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow(["Eval", "m", "p", "t", "Cl", "Cd", "Error"])
            writer.writerows(objective_function.history)
            
        # DAT Export
        dat_path = os.path.join(output_dir, f"airfoil_NACA_{naca_opt_str}.dat")
        save_airfoil_to_file(X, Y, dat_path)
            
        print(f"\n[+] DATA EXPORTED: Results, plot, and airfoil .dat saved in '{folder_name}'")
        # -----------------------------------------------

        print(f"Status: Completed (Iterations: {result.nit}, Evaluations: {result.nfev})")
        print(f"Final error score: {result.fun:.6f}")
        
        print("\nACHIEVED PERFORMANCES:")
        if final_cl is not None and final_cd is not None:
            if abs(final_alpha - TARGET_ALPHA) > 0.1:
                print(f"Cl: {final_cl:.4f} (Obtained at stall at {final_alpha}° instead of {TARGET_ALPHA}°)")
                print(f"Cd: {final_cd:.5f}")
                print("WARNING: The airfoil stalls before reaching the requested angle.")
            else:
                print(f"Cl: {final_cl:.4f} (Target: {TARGET_CL})")
                print(f"Cd: {final_cd:.5f} (Limit: {MAX_CD})")
                if final_cd > MAX_CD:
                    print("Note: the final Cd exceeds the imposed limit for the requested Cl.")
        else:
            print("Cl: Data not available (Separated or non-converging flow)")
            print("Cd: Data not available (Separated or non-converging flow)")
            
        print("\nFOUND GEOMETRY:")
        print(f"Maximum camber (m): {optimal_params[0]:.4f}")
        print(f"Camber position (p): {optimal_params[1]:.4f}")
        print(f"Maximum thickness (t): {optimal_params[2]:.4f}")
        print(f"Approximated NACA Airfoil: NACA {naca_opt_str}")
        print("======================================================================\n")
        
        plt.figure(figsize=(12, 6))
        plt.plot(coords_optimal[0], coords_optimal[1], 'k-', linewidth=2, label=f'NACA Airfoil {naca_opt_str}')
        plt.plot(coords_optimal[2], coords_optimal[3], 'k-', linewidth=2)
        
        if final_cl is not None and final_cd is not None:
            if abs(final_alpha - TARGET_ALPHA) > 0.1:
                titolo = f"Result (Stall at {final_alpha}°): Cl={final_cl:.3f} | Cd={final_cd:.4f} (Re={TARGET_REYNOLDS:.1f})"
            else:
                titolo = f"Optimization Result: Cl={final_cl:.3f} | Cd={final_cd:.4f} (Re={TARGET_REYNOLDS:.1f}, Alpha={TARGET_ALPHA}°)"
        else:
            titolo = f"Optimization Result: Non-converging parameters (Re={TARGET_REYNOLDS:.1f}, Alpha={TARGET_ALPHA}°)"
            
        plt.title(titolo)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot in the results folder (at 300 dpi for high resolution)
        plot_path = os.path.join(output_dir, f"plot_NACA_{naca_opt_str}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    else:
        print(f"Optimization failed: {result.message}")