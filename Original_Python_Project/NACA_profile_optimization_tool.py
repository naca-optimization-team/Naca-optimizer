import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from scipy.optimize import minimize
import shlex
import shutil

# ==============================================================================
# FUNZIONI DI GENERAZIONE DEL PROFILO ALARE
# ==============================================================================

def naca4(m_param, p_param, t_param, c=1.0, n=100):
    """
    Genera le coordinate di un profilo alare NACA a 4 cifre da parametri.
    :param m_param: Curvatura massima (es. 0.02 per il 2%)
    :param p_param: Posizione della curvatura massima (es. 0.4 per il 40%)
    :param t_param: Spessore massimo (es. 0.12 per il 12%)
    :param c: Corda del profilo
    :param n: Numero di punti per semi-profilo
    """
    x = np.linspace(0, c, n)
    
    # Distribuzione dello spessore
    yt = 5 * t_param * c * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

    if p_param == 0 or m_param == 0:
        # Profilo simmetrico
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
        # Profilo curvo
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

    # Unisci le coordinate per il file di XFOIL (in senso orario, partendo dal bordo d'uscita)
    X = np.concatenate((np.flip(xu), xl[1:]))
    Y = np.concatenate((np.flip(yu), yl[1:]))
    
    return X, Y, (xu, yu, xl, yl)


def save_airfoil_to_file(X, Y, filename):
    """Salva le coordinate del profilo in un file .dat per XFOIL."""
    with open(filename, "w") as f:
        for i in range(len(X)):
            f.write(f"{X[i]:.6f} {Y[i]:.6f}\n")

# ==============================================================================
# FUNZIONE DI ANALISI CFD (WRAPPER XFOIL)
# ==============================================================================

def run_xfoil_analysis(airfoil_file, alpha, Re, Mach=0.0):
    """
    Esegue un'analisi XFOIL per un dato profilo e condizioni di volo.
    Restituisce (CL, CD).
    """
    xfoil_input_file = "xfoil_input.in"
    polar_file = "polar.dat"

    # Crea il file di comandi per XFOIL
    with open(xfoil_input_file, "w") as f:
        f.write(f"LOAD {airfoil_file}\n")
        f.write("PANE\n")  # Raffina la discretizzazione
        f.write("OPER\n")
        f.write(f"Visc {Re}\n")
        f.write(f"Mach {Mach}\n")
        f.write("PACC\n")
        f.write(f"{polar_file}\n\n") # File di output per i dati polari
        f.write(f"ALFA {alpha}\n")
        f.write("\n")      # Esci dal menu OPER
        f.write("QUIT\n")

    # --- MODIFICA 1: Percorso assoluto per xfoil.exe ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    # Esegui XFOIL con i percorsi protetti da virgolette (per evitare problemi con gli spazi nei nomi delle cartelle)
    command = f'"{xfoil_exe_path}" < "{xfoil_input_file}"'
    # ---------------------------------------------------

    try:
        # Impostiamo un timeout per evitare che XFOIL si blocchi all'infinito se non converge
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"!!! ERRORE: XFOIL non è riuscito a completare l'analisi per {airfoil_file}.")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"    Output di errore da XFOIL:\n{e.stderr}")
        else:
            print(f"    Dettagli errore: {e}")
        return None, None

    # Leggi i risultati dal file polare
    cl, cd = None, None
    try:
        with open(polar_file, "r") as f:
            lines = [line for line in f if not line.startswith("#")]
            if lines:
                data = lines[-1].split()
                if len(data) >= 3:
                    cl = float(data[1])
                    cd = float(data[2])
    except (IOError, IndexError):
        print(f"Avviso: Impossibile leggere il file di output {polar_file}.")
        pass
    
    # Pulisci i file temporanei
    for f in [xfoil_input_file, polar_file, airfoil_file]:
        if os.path.exists(f):
            os.remove(f)
            
    return cl, cd

# ==============================================================================
# FUNZIONE OBIETTIVO PER L'OTTIMIZZAZIONE
# ==============================================================================

def objective_function(params, Re, alpha):
    """
    Funzione obiettivo da minimizzare.
    params: [m_param, p_param, t_param]
    Restituisce -CL/CD (minimizzare -X è come massimizzare X).
    """
    m, p, t = params
    
    # Vincoli impliciti: i parametri devono avere senso
    if not (0.0 <= m < 0.1 and 0.1 <= p < 0.8 and 0.05 <= t < 0.25):
        return 1e6 # Penalità alta per parametri non validi

    airfoil_name = f"temp_naca.dat"
    
    # 1. Genera profilo e salvalo
    X, Y, _ = naca4(m, p, t)
    save_airfoil_to_file(X, Y, airfoil_name)
    
    # 2. Esegui analisi
    cl, cd = run_xfoil_analysis(airfoil_name, alpha, Re)
    
    # 3. Calcola valore obiettivo
    if cl is not None and cd is not None and cd > 0:
        efficiency = cl / cd
        print(f"Test NACA(m={m:.3f}, p={p:.3f}, t={t:.3f}) -> CL/CD = {efficiency:.2f}")
        return -efficiency  # Minimizziamo il negativo dell'efficienza
    else:
        # Rendi esplicito il motivo della penalità
        print(f"--- Analisi fallita per NACA(m={m:.3f}, p={p:.3f}, t={t:.3f}). Assegno penalità.")
        # Penalità se l'analisi fallisce o dà risultati assurdi
        return 1e6

# ==============================================================================
# BLOCCO PRINCIPALE DI ESECUZIONE
# ==============================================================================

if __name__ == "__main__":
    
    # --- MODIFICA 2: CONTROLLO PREREQUISITI AGGIORNATO ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xfoil_exe_path = os.path.join(script_dir, "xfoil.exe")
    
    if not os.path.exists(xfoil_exe_path):
        print("="*60)
        print("!!! ERRORE CRITICO: Eseguibile 'xfoil.exe' non trovato.")
        print(f"Assicurati che XFOIL sia presente in questa cartella:\n{script_dir}")
        print("Lo script non può funzionare senza XFOIL.")
        print("="*60)
        exit()
    # -----------------------------------------------------

    # --- CONDIZIONI DI VOLO E PARAMETRI DI OTTIMIZZAZIONE ---
    TARGET_REYNOLDS = 1_000_000
    TARGET_ALPHA = 1

    # Profilo iniziale (NACA 0012 - simmetrico, una buona sfida per l'ottimizzatore)
    # L'ottimizzatore dovrà "imparare" ad aggiungere curvatura (m > 0) per migliorare l'efficienza.
    # m=0%, p=40% (irrilevante se m=0), t=12%
    initial_guess = [0.0, 0.4, 0.12]

    # Limiti per i parametri [m, p, t] durante l'ottimizzazione
    # m: 0% -> 6%
    # p: 30% -> 50%
    # t: 8% -> 18%
    bounds = [(0.0, 0.06), (0.3, 0.5), (0.08, 0.18)]

    print("--- OTTIMIZZAZIONE PROFILO ALARE ---")
    print(f"Condizioni: Re = {TARGET_REYNOLDS}, Angolo d'attacco = {TARGET_ALPHA}°")
    print(f"Profilo iniziale (m,p,t): {initial_guess} -> NACA 0012 Simmetrico")
    print("-" * 35)

    # Esegui l'ottimizzazione
    # Aumentiamo maxiter per dare più tempo all'algoritmo e aggiungiamo una tolleranza sulla funzione obiettivo (ftol)
    result = minimize(
        objective_function,
        initial_guess,
        args=(TARGET_REYNOLDS, TARGET_ALPHA),
        method='SLSQP',  # Un buon algoritmo per problemi con vincoli (bounds)
        bounds=bounds,
        options={'disp': True, 'maxiter': 150, 'ftol': 1e-8} # Aumentato maxiter e aggiunta tolleranza
    )

    print("-" * 35)
    print("--- RIEPILOGO OTTIMIZZAZIONE ---")
    print(f"Messaggio di terminazione: {result.message}")
    print(f"Numero di iterazioni: {result.nit}")
    print(f"Numero di valutazioni funzione: {result.nfev}")
    
    if result.success and -result.fun > 0:
        optimal_params = result.x
        max_efficiency = -result.fun
        print(f"\nOttimizzazione completata con successo!")
        print(f"Efficienza massima (CL/CD) trovata: {max_efficiency:.2f}")
        print(f"Parametri ottimali (m, p, t): {optimal_params[0]:.4f}, {optimal_params[1]:.4f}, {optimal_params[2]:.4f}")
        
        # Visualizza i profili
        _, _, coords_initial = naca4(*initial_guess)
        _, _, coords_optimal = naca4(*optimal_params)

        plt.figure(figsize=(12, 8))
        plt.plot(coords_initial[0], coords_initial[1], 'b--', label='Iniziale (NACA 0012 Simmetrico)')
        plt.plot(coords_initial[2], coords_initial[3], 'b--')
        plt.plot(coords_optimal[0], coords_optimal[1], 'r-', label=f'Ottimale (CL/CD={max_efficiency:.2f})')
        plt.plot(coords_optimal[2], coords_optimal[3], 'r-')
        
        plt.title('Confronto tra Profilo Iniziale e Ottimale')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    else:
        print("\nL'ottimizzazione non è riuscita a trovare una soluzione migliorativa o è fallita.")
        print(f"Causa: {result.message}")
        print(f"Ultimi parametri tentati: {result.x}")
        print(f"Valore funzione obiettivo finale: {result.fun}")