import os
import time
import argparse
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq
from pycbc.waveform import get_td_waveform
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, griddata

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
    """Generates a Planck-taper window.

    Planck-taper window is a window function that is flat in the middle and
    smoothly tapers to zero at both ends.

    The shape of the tapered sections is derived from a function related to
    Planck's law, which provides an infinitely differentiable transition from the
    flat-top region (with a value of 1) to the zero-value regions. This
    implementation uses the logistic function (`scipy.special.expit`) for a
    numerically stable computation of the taper.

    Parameters
    ----------
    N : int
        The total number of points in the output window.
    epsilon : float, optional
        The fraction of the window's length at **each end** that is tapered.
        This value must be in the range (0, 0.5). For example, if `N` is
        1000 and `epsilon` is 0.1, the first 100 points and the last 100
        points form the tapered sections. The default is 0.1.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array of shape `(N,)` containing the window values, which
        range from 0.0 to 1.0.

    Raises
    ------
    ValueError
        If `epsilon` is not in the valid range of (0, 0.5).
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

def generate_fd_waveform(params, M_total, f_lower, delta_t, window_type='lal_planck', padding_type='power_of_2', epsilon=0.1, num_extrema_start=32, num_extrema_end=32):
    q = params['q']
    chi = params['chi']
    m2 = M_total / (1 + q)
    m1 = q * M_total / (1 + q)


    hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                                mass1=m1, mass2=m2,
                                spin1z=chi, spin2z=chi,
                                delta_t=delta_t,
                                f_lower=f_lower)
    h_td_raw = hp.numpy()

    if window_type == "tukey":
        window = windows.tukey(len(h_td_raw), alpha=0.1)
    elif window_type == "planck":
        window = planck_taper(len(h_td_raw), epsilon=epsilon)
    elif window_type == "lal_planck":
        window = Planck_window_LAL(h_td_raw, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end) 
    else:
        window = np.ones(len(h_td_raw))

    h_td_windowed = h_td_raw * window
    
    L = len(h_td_windowed)
    
    if padding_type == "power_of_2":
      nfft = 2 ** int(np.ceil(np.log2(2 * L)))
    elif padding_type == 'double':
      nfft = 2 * L

    h_td = np.pad(h_td_windowed, (0, nfft - L))

    # if padding_type == "power_of_2":
    #     nfft = 2 ** int(np.ceil(np.log2(2 * L)))
    # elif padding_type == "double":
    #     nfft = 2 * L

    # total_pad = nfft - L
    # pad_left = total_pad // 2
    # pad_right = total_pad - pad_left

    # h_td = np.pad(h_td_windowed, (pad_left, pad_right))

    freqs = rfftfreq(nfft, delta_t)
    h_fd = rfft(h_td)
    return freqs, h_fd

def get_amp_phase(freqs, h_fd):
    amp = np.abs(h_fd)
    phase = np.unwrap(np.angle(h_fd))

    poly_fit = np.polyfit(freqs, phase, 1)
    linear_phase = np.polyval(poly_fit, freqs)
    phase_centered = phase - linear_phase
    
    phase_offset = phase_centered[0]
    phase_anchored = phase_centered - phase_offset

    return amp, phase_anchored

def generate_sparse_grid(f_min, f_max, num_points, power=4/3):
    if power == 1:
        return np.geomspace(f_min, f_max, num_points)

    y_min = f_min**(1 - power)
    y_max = f_max**(1 - power)

    y_grid = np.linspace(y_min, y_max, num_points)

    f_grid = y_grid**(1 / (1 - power))

    f_grid[0] = f_min
    f_grid[-1] = f_max
    
    return f_grid

# -----------------------------------------------------------------------------
# ## Step I: Generate and Pre-process Training Waveforms
# -----------------------------------------------------------------------------
print("Step I: Generating training data...")

bounds_q = (1, 10)
bounds_chi = (-1, 1)

q_vals = np.concatenate((
    np.linspace(bounds_q[0], bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.2, 20, endpoint=False),
    np.linspace(bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.2, bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.8, 10, endpoint=False),
    np.linspace(bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.8, bounds_q[1], 20)
))

chi_vals = np.concatenate((
    np.linspace(bounds_chi[0], bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.2, 20, endpoint=False),
    np.linspace(bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.2, bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.8, 10, endpoint=False),
    np.linspace(bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.8, bounds_chi[1], 20)
))

param_grid_q, param_grid_chi = np.meshgrid(q_vals, chi_vals)
params_list = [{'q': q, 'chi': chi} for q, chi in zip(param_grid_q.flatten(), param_grid_chi.flatten())]

f_lower = 16.0
delta_t = 1/4096
f_min_mask = 20.0
f_max_mask = 725.0
window_type = "planck"
epsilon = 0.1
num_extrema_start=32
num_extrema_end=32

q_fixed = 5.0
chi_fixed = 0.0
n_trials = 5

true_times = []
surr_times = []
speedups = []
M_total_list = np.arange(20, 401, 20)

for M_total in M_total_list:
    
    raw_amps = []
    raw_phases = []
    raw_freqs = []
    amp_norms = []
    valid_params = []

    print(f"\n-- Working on M_total = {M_total} M_sun --")
    for params in params_list:
        freqs, h_fd = generate_fd_waveform(params=params, M_total=M_total, f_lower=f_lower, delta_t=delta_t, window_type=window_type, padding_type='power_of_2', epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end)

        mask = (freqs >= f_min_mask) & (freqs <= f_max_mask)
        freqs_masked = freqs[mask]

        amp, phase = get_amp_phase(freqs_masked, h_fd[mask])

        df = freqs_masked[1] - freqs_masked[0]
        norm = np.sqrt(np.sum(amp**2) * df)

        raw_amps.append(amp / norm)
        amp_norms.append(norm)
        raw_phases.append(phase)
        raw_freqs.append(freqs_masked)
        valid_params.append(params)

    print(f"Generated {len(raw_amps)} valid waveforms for M_total={M_total}.")

    # -----------------------------------------------------------------------------
    # ## Step II: Define Sparse Frequency Grids and Interpolate
    # -----------------------------------------------------------------------------
    f_min_grid = 20.0
    f_max_grid = 725.0

    sparse_freq_amp = generate_sparse_grid(f_min_grid, f_max_grid, num_points=200, power=1.4) 
    sparse_freq_phase = generate_sparse_grid(f_min_grid, f_max_grid, num_points=200, power=4/3)

    A_mat = np.zeros((len(sparse_freq_amp), len(raw_amps)))
    Phi_mat = np.zeros((len(sparse_freq_phase), len(raw_phases)))

    for i, (amp, phase, freqs) in enumerate(zip(raw_amps, raw_phases, raw_freqs)):

        spline_amp = UnivariateSpline(freqs, amp, s=0, k=3, ext=2)
        spline_phase = UnivariateSpline(freqs, phase, s=0, k=3, ext=2)

        fmin, fmax = freqs[0], freqs[-1]

        A_mat[:, i] = spline_amp(sparse_freq_amp)
        Phi_mat[:, i] = spline_phase(sparse_freq_phase)

    # -----------------------------------------------------------------------------
    # ## Step III: Compute Reduced Bases via SVD
    # -----------------------------------------------------------------------------
    Ua, sa, Vta = np.linalg.svd(A_mat, full_matrices=False)
    Up, sp, Vtp = np.linalg.svd(Phi_mat, full_matrices=False)

    rank_a = 30
    rank_p = 20

    B_a = Ua[:, :rank_a] 
    B_p = Up[:, :rank_p]

    # -----------------------------------------------------------------------------
    # ## Step IV: Interpolate Projection Coefficients
    # -----------------------------------------------------------------------------
    Ca = B_a.T @ A_mat
    Cp = B_p.T @ Phi_mat

    q_unique = np.unique(param_grid_q)
    chi_unique = np.unique(param_grid_chi)

    interpolants_a = []
    for i in range(rank_a):
        coeff_grid = Ca[i, :].reshape(len(chi_unique), len(q_unique))
        interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
        interpolants_a.append(interp)

    interpolants_p = []
    for i in range(rank_p):
        coeff_grid = Cp[i, :].reshape(len(chi_unique), len(q_unique))
        interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
        interpolants_p.append(interp)

    amp_norms_grid = np.array(amp_norms).reshape(len(chi_unique), len(q_unique))
    interp_amp_norm = RectBivariateSpline(chi_unique, q_unique, amp_norms_grid, kx=3, ky=3)

    # -----------------------------------------------------------------------------
    # ## Step V: Assemble and Evaluate the Surrogate Model
    # -----------------------------------------------------------------------------
    def evaluate_surrogate_fd(q_star, chi_star, freqs_out):
        ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_a])
        cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_p])

        amp_recon_sparse = B_a @ ca_star
        phase_recon_sparse = B_p @ cp_star

        spline_amp = UnivariateSpline(sparse_freq_amp, amp_recon_sparse, s=0, k=3, ext=2)
        spline_phase = UnivariateSpline(sparse_freq_phase, phase_recon_sparse, s=0, k=3, ext=2)

        amp_final = spline_amp(freqs_out)
        phase_final = spline_phase(freqs_out)

        norm_star = interp_amp_norm(chi_star, q_star)[0, 0]
        amp_final *= norm_star

        h_fd_recon = amp_final * np.exp(1j * phase_final)
        
        return freqs_out, h_fd_recon

    print("\nRunning speed calculation loop...")

    # -----------------------------------------------------------------------------
    # ## Run a test case to validate the surrogate model
    # -----------------------------------------------------------------------------
    test_params = {'q': q_fixed, 'chi': chi_fixed}

    start = time.time()
    true_freqs = None
    true_h_fd = None
    for _ in range(n_trials):
        true_freqs, true_h_fd = generate_fd_waveform(test_params, M_total, f_lower, delta_t, window_type=window_type, padding_type='power_of_2', epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end)

    end = time.time()
    true_time = (end - start) / n_trials
    true_times.append(true_time)
    print(f"Time taken by SEOBNRv4_opt: {true_time:.3f}")

    mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
    true_freqs_masked = true_freqs[mask]
    true_h_fd_masked = true_h_fd[mask]

    start = time.time()
    surr_freqs = None
    surr_h_fd = None
    for _ in range(n_trials):
        surr_freqs, surr_h_fd = evaluate_surrogate_fd(q_fixed, chi_fixed, true_freqs_masked)
    end = time.time()
    surr_time = (end - start) / n_trials
    surr_times.append(surr_time)
    print(f"Time taken by Surrogate Model: {surr_time:.3f}")

    speedup = true_time / surr_time if (surr_time != 0 and np.isfinite(surr_time)) else np.nan
    speedups.append(speedup)

# ----------------------------------------------------
# Plot absolute runtimes
# ----------------------------------------------------
os.makedirs(f"{args.results_dir}", exist_ok=True)

plt.figure(figsize=(8,6))
plt.plot(M_total_list, true_times, marker="o", lw=2, label="True Model")
plt.plot(M_total_list, surr_times, marker="o", lw=2, label="Surrogate Model")
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
plt.plot(M_total_list, speedups, marker="o", lw=2)
plt.xlabel("Total Mass $M_{tot} \\, (M_\\odot)$", fontsize=12)
plt.ylabel("Speedup (True / Surrogate)", fontsize=12)
plt.title(f"Surrogate Speedup vs. True Model (q={q_fixed}, χ={chi_fixed})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.savefig(f"{args.results_dir}/Speedup_vs_Mass.pdf")
plt.show()
