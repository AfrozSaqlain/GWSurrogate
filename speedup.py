import os
import time
import pickle
import argparse
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq
from pycbc.waveform import get_td_waveform
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

parser = argparse.ArgumentParser(description="Train a surrogate model for gravitational waveforms.")
parser.add_argument('--results-dir', type=str, default='Results', help='Directory to save results and plots.')
args = parser.parse_args()

if args.results_dir != 'Results' and not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# -----------------------------------------------------------------------------
# ## Setup and Helper Functions
# -----------------------------------------------------------------------------
LALSIMULATION_RINGING_EXTENT = 19
def Planck_window_LAL(data, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=2, num_extrema_end=2):
    """
    Parameters:
    -----------
    data: 1D numpy array 
        data to taper
    taper_method: string
        Tapering method. Available methods are: 
        "LAL_SIM_INSPIRAL_TAPER_START"
        "LAL_SIM_INSPIRAL_TAPER_END"
        "LAL_SIM_INSPIRAL_TAPER_STARTEND"
    num_extrema_start: int
        number of extrema till which to taper from the start
    num_extrema_end: int
        number of extrema till which to taper from the end
        
    Returns:
    --------
    window: 1D numpy array
        Planck tapering window
    """
    start=0
    end=0
    n=0
    length = len(data)

    # Search for start and end of signal
    flag = 0
    i = 0
    while(flag == 0 and i < length):
        if (data[i] != 0.):
            start = i
            flag = 1
        i+=1
    if (flag == 0):
        raise ValueError("No signal found in the vector. Cannot taper.\n")

    flag = 0
    i = length - 1
    while( flag == 0 ):
        if( data[i] != 0. ):
                end = i
                flag = 1
        i-=1

    # Check we have more than 2 data points 
    if( (end - start) <= 1 ):
        raise RuntimeError( "Data less than 3 points, cannot taper!\n" )

    # Calculate middle point in case of short waveform
    mid = int((start+end)/2)

    window = np.ones(length)
    # If requested search for num_extrema_start-th peak from start and taper
    if( taper_method != "LAL_SIM_INSPIRAL_TAPER_END" ):
        flag = 0
        i = start+1
        while ( flag < num_extrema_start and i != mid ):
            if( abs(data[i]) >= abs(data[i-1]) and
                abs(data[i]) >= abs(data[i+1]) ):
            
                if( abs(data[i]) == abs(data[i+1]) ):
                    i+=1
                # only count local extrema more than 19 samples in
                if ( i-start > LALSIMULATION_RINGING_EXTENT ):
                    flag+=1
                n = i - start
            i+=1

        # Have we reached the middle without finding `num_extrema_start` peaks?
        if( flag < num_extrema_start ):
            n = mid - start
            print(f"""WARNING: Reached the middle of waveform without finding {num_extrema_start} extrema. Tapering only till the middle from the beginning.""")

        # Taper to that point
        realN = n
        window[:start+1] = 0.0
        realI = np.arange(1, n - 1)
        z = (realN - 1.0)/realI + (realN - 1.0)/(realI - (realN - 1.0))
        window[start+1: start+n-1] = 1.0/(np.exp(z) + 1.0)

    # If requested search for num_extrema_end-th peak from end
    if( taper_method == "LAL_SIM_INSPIRAL_TAPER_END" or taper_method == "LAL_SIM_INSPIRAL_TAPER_STARTEND" ):
        i = end - 1
        flag = 0
        while( flag < num_extrema_end and i != mid ):
            if( abs(data[i]) >= abs(data[i+1]) and
                abs(data[i]) >= abs(data[i-1]) ):
                if( abs(data[i]) == abs(data[i-1]) ):
                    i-=1
                # only count local extrema more than 19 samples in
                if ( end-i > LALSIMULATION_RINGING_EXTENT ):
                    flag+=1
                n = end - i
            i-=1

        # Have we reached the middle without finding `num_extrema_end` peaks?
        if( flag < num_extrema_end ):
            n = end - mid
            print(f"""WARNING: Reached the middle of waveform without finding {num_extrema_end} extrema. Tapering only till the middle from the end.""")

        # Taper to that point
        realN = n
        window[end:] = 0.0        
        realI = -np.arange(-n+2, 0)
        z = (realN - 1.0)/realI + (realN - 1.0)/(realI - (realN - 1.0))
        window[end-n+2:end] = 1.0/(np.exp(z) + 1.0)

    return window

def planck_taper(N, epsilon=0.1):
    """
    Planck-taper window.
    """
    if not (0 < epsilon < 0.5):
        raise ValueError("epsilon must be between 0 and 0.5")

    w = np.ones(N)
    L = int(epsilon * N)

    if L == 0:
        return w

    n = np.arange(1, L)
    x = L/n - L/(L-n)
    w[:L-1] = expit(-x)
    w[0] = 0.0

    x = L/(L-n) - L/n
    w[-(L-1):] = expit(-x)
    w[-1] = 0.0

    return w

# ----------------------------------------------------
# Load surrogate model
# ----------------------------------------------------
def load_surrogate(filename):
    """Load surrogate model and rebuild interpolants."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    interpolants_a = []
    for i in range(data["Ca"].shape[0]):
        coeff_grid = data["Ca"][i, :].reshape(len(data["chi_unique"]), len(data["q_unique"]))
        interp = RectBivariateSpline(data["chi_unique"], data["q_unique"], coeff_grid, kx=3, ky=3)
        interpolants_a.append(interp)

    interpolants_p = []
    for i in range(data["Cp"].shape[0]):
        coeff_grid = data["Cp"][i, :].reshape(len(data["chi_unique"]), len(data["q_unique"]))
        interp = RectBivariateSpline(data["chi_unique"], data["q_unique"], coeff_grid, kx=3, ky=3)
        interpolants_p.append(interp)

    interp_amp_norm = RectBivariateSpline(data["chi_unique"], data["q_unique"], data["amp_norms_grid"], kx=3, ky=3)

    return {
        "sparse_freq_amp": data["sparse_freq_amp"],
        "sparse_freq_phase": data["sparse_freq_phase"],
        "B_a": data["B_a"],
        "B_p": data["B_p"],
        "interpolants_a": interpolants_a,
        "interpolants_p": interpolants_p,
        "interp_amp_norm": interp_amp_norm
    }

# ----------------------------------------------------
# Surrogate evaluation function
# ----------------------------------------------------
def evaluate_surrogate_fd(q_star, chi_star, freqs_out, surrogate):
    ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in surrogate["interpolants_a"]])
    cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in surrogate["interpolants_p"]])

    amp_recon_sparse = surrogate["B_a"] @ ca_star
    phase_recon_sparse = surrogate["B_p"] @ cp_star

    spline_amp = UnivariateSpline(surrogate["sparse_freq_amp"], amp_recon_sparse, s=0, k=3, ext=0)
    spline_phase = UnivariateSpline(surrogate["sparse_freq_phase"], phase_recon_sparse, s=0, k=3, ext=0)

    amp_final = spline_amp(freqs_out)
    phase_final = spline_phase(freqs_out)

    norm_star = surrogate["interp_amp_norm"](chi_star, q_star)[0, 0]
    amp_final *= norm_star

    return freqs_out, amp_final * np.exp(1j * phase_final)

# ----------------------------------------------------
# Load trained surrogate
# ----------------------------------------------------
surrogate = load_surrogate("Models/surrogate_model.pkl")
print("Surrogate model loaded successfully!")

# ----------------------------------------------------
# Benchmark setup
# ----------------------------------------------------
f_lower = 20.0
f_min_grid = 25.0
f_max_grid = 1024.0
delta_t = 1/4096
nfft = 16 * 4096

masses = np.linspace(10, 400, 100) 
q_fixed = 5.0
chi_fixed = 0.0
n_trials = 1  

true_times = []
surr_times = []
speedups = []

# ----------------------------------------------------
# Benchmark loop
# ----------------------------------------------------
print("\nBenchmarking speedup for varying total masses...")

window_type= 'lal_planck'  # Options: 'tukey', 'planck', 'lal_planck'

for M_total in masses:
    m2 = M_total / (1 + q_fixed)
    m1 = q_fixed * M_total / (1 + q_fixed)

    t_true_runs, t_surr_runs = [], []

    for _ in range(n_trials):
        # --- True waveform timing ---
        start_true = time.time()
        try:
            hp, _ = get_td_waveform(approximant="SEOBNRv4_opt",
                                    mass1=m1, mass2=m2,
                                    spin1z=chi_fixed, spin2z=chi_fixed,
                                    delta_t=delta_t,
                                    f_lower=f_lower)
            h_td_raw = hp.numpy()

            if window_type == "tukey":
                window = windows.tukey(len(h_td_raw), alpha=0.1)
            elif window_type == "planck":
                window = planck_taper(len(h_td_raw), epsilon=0.1)
            elif window_type == "lal_planck":
                window = Planck_window_LAL(h_td_raw, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=2, num_extrema_end=2) 
            else:
                window = np.ones(len(h_td_raw))
            
            h_td_windowed = h_td_raw * window

            L = len(h_td_windowed)
            nfft = 2 ** int(np.ceil(np.log2(2 * L)))

            h_td = np.pad(h_td_windowed, (0, nfft - L))

            freqs_true = rfftfreq(nfft, delta_t)
            h_fd_true = rfft(h_td)

            mask_true = (freqs_true >= f_min_grid) & (freqs_true <= f_max_grid)
            freqs_true = freqs_true[mask_true]
            h_fd_true = h_fd_true[mask_true]
        except Exception as e:
            print(f"Failed true waveform at M={M_total}: {e}")
            continue
        end_true = time.time()
        t_true_runs.append(end_true - start_true)

        # --- Surrogate timing ---
        start_surr = time.time()
        _, h_fd_surr = evaluate_surrogate_fd(q_fixed, chi_fixed, freqs_true, surrogate)
        end_surr = time.time()
        t_surr_runs.append(end_surr - start_surr)

    if not t_true_runs or not t_surr_runs:
        continue

    t_true_avg = np.mean(t_true_runs)
    t_surr_avg = np.mean(t_surr_runs)

    true_times.append(t_true_avg)
    surr_times.append(t_surr_avg)
    speedups.append(t_true_avg / t_surr_avg if t_surr_avg > 0 else np.nan)

    print(f"M={M_total:.1f}: True={t_true_avg:.3f}s, Surrogate={t_surr_avg:.3e}s, "
          f"Speedup={t_true_avg/t_surr_avg:.1f}x (averaged over {n_trials} runs)")

# ----------------------------------------------------
# Plot absolute runtimes
# ----------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(masses[:len(true_times)], true_times, marker="o", lw=2, label="True Model")
plt.plot(masses[:len(surr_times)], surr_times, marker="s", lw=2, label="Surrogate Model")
plt.xlabel("Total Mass $M_{tot} \\, (M_\\odot)$", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.title(f"Runtime Comparison (q={q_fixed}, χ={chi_fixed})", fontsize=14)
plt.legend()
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.savefig(f"{args.results_dir}/Runtime_vs_Mass.pdf")
plt.show()

# ----------------------------------------------------
# Plot speedup
# ----------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(masses[:len(speedups)], speedups, marker="o", lw=2, color="purple")
plt.xlabel("Total Mass $M_{tot} \\, (M_\\odot)$", fontsize=12)
plt.ylabel("Speedup (True / Surrogate)", fontsize=12)
plt.title(f"Surrogate Speedup vs. True Model (q={q_fixed}, χ={chi_fixed})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.savefig(f"{args.results_dir}/Speedup_vs_Mass.pdf")
plt.show()
