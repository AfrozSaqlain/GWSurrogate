import os
import pycbc
import pickle
import argparse
import pycbc.psd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.signal import windows
from mpl_toolkits.mplot3d import Axes3D
from pycbc.waveform import get_td_waveform
from scipy.fft import rfft, rfftfreq, irfft
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
def Planck_window_LAL(data, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=32, num_extrema_end=32):
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

def generate_fd_waveform(params, f_lower, delta_t, window_type='lal_planck', epsilon=0.1):
    """Generates a time-domain waveform, applies a window, and converts to frequency domain."""
    q = params['q']
    chi = params['chi']
    M_total = 40.0
    m2 = M_total / (1 + q)
    m1 = q * M_total / (1 + q)

    try:
        hp, _ = get_td_waveform(approximant='SEOBNRv4_opt',
                                  mass1=m1, mass2=m2,
                                  spin1z=chi, spin2z=chi,
                                  delta_t=delta_t,
                                  f_lower=f_lower)
        h_td_raw = hp.numpy()
    except Exception as e:
        print(f"Could not generate waveform for q={q}, chi={chi}: {e}")
        return None, None

    if window_type == "tukey":
        window = windows.tukey(len(h_td_raw), alpha=0.1)
    elif window_type == "planck":
        window = planck_taper(len(h_td_raw), epsilon=epsilon)
    elif window_type == "lal_planck":
        window = Planck_window_LAL(h_td_raw, taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', num_extrema_start=2, num_extrema_end=2) 
    else:
        window = np.ones(len(h_td_raw))

    h_td_windowed = h_td_raw * window
    
    #refined the zero-padding scheme by extending the waveform length to the smallest power of two greater than twice its original length. This approach ensures efficient FFT computation and improves frequency resolution by providing a denser sampling of the frequency axis without altering the underlying physical content of the signal.

    L = len(h_td_windowed)
    nfft = 2 ** int(np.ceil(np.log2(2 * L)))

    h_td = np.pad(h_td_windowed, (0, nfft - L))

    freqs = rfftfreq(nfft, delta_t)
    h_fd = rfft(h_td)
    return freqs, h_fd

def get_amp_phase(freqs, h_fd):
    """
    Extracts amplitude and unwrapped phase.
    Perform a two-step centering:
    1. Removes a linear fit to detrend the phase.
    2. Anchors the phase to start at zero for numerical stability.
    """
    amp = np.abs(h_fd)
    phase = np.unwrap(np.angle(h_fd))

    if len(freqs) > 1:
        poly_fit = np.polyfit(freqs, phase, 1)
        linear_phase = np.polyval(poly_fit, freqs)
        phase_centered = phase - linear_phase
    else:
        phase_centered = phase
    if len(phase_centered) > 0:
        phase_offset = phase_centered[0]
        phase_anchored = phase_centered - phase_offset
    else:
        phase_anchored = phase_centered

    return amp, phase_anchored

def generate_sparse_grid(f_min, f_max, num_points, power=4/3):
    """
    Implements a simplified version of the 'Constant Spline Error' (CSE) grid generation
    The grid spacing Delta(f) is proportional to f^power.
    """
    if power == 1:
        return np.geomspace(f_min, f_max, num_points)
    
    def integrated_spacing(f):
        return f**(1-power)

    C = (integrated_spacing(f_max) - integrated_spacing(f_min)) / (num_points - 1)

    grid = []
    f_current = f_min
    for _ in range(num_points):
        grid.append(f_current)
        if f_current >= f_max:
            break
        f_next_integrated = integrated_spacing(f_current) + C
        f_current = f_next_integrated**(1 / (1-power))
        
    grid = np.array(grid)
    return grid

# -----------------------------------------------------------------------------
# ## Step I: Generate and Pre-process Training Waveforms
# -----------------------------------------------------------------------------
print("Step I: Generating training data...")

q_vals = np.linspace(1, 10, 30)
chi_vals = np.linspace(-1.0, 1.0, 30)
param_grid_q, param_grid_chi = np.meshgrid(q_vals, chi_vals)
params_list = [{'q': q, 'chi': chi} for q, chi in zip(param_grid_q.flatten(), param_grid_chi.flatten())]

f_lower = 20.0
f_min_grid = 25.0
f_max_grid = 1024.0
delta_t = 1/4096
# nfft = 16 * 4096

window_type = "planck"  # Options: "tukey", "planck", "lal_planck", or None

raw_amps = []
raw_phases = []
raw_freqs = []
amp_norms = []
valid_params = []

for params in params_list:
    freqs, h_fd = generate_fd_waveform(params, f_lower, delta_t, window_type=window_type, epsilon=0.2)
    if freqs is None: continue

    mask = (freqs >= 24) & (freqs <= 1025)
    freqs_masked = freqs[mask]
    
    if freqs_masked.size == 0:
        continue

    amp, phase = get_amp_phase(freqs_masked, h_fd[mask])

    if amp.size < 2:
        continue
    df = freqs_masked[1] - freqs_masked[0]

    norm = np.sqrt(np.sum(amp**2) * df)

    if norm == 0 or not np.isfinite(norm):
        continue

    raw_amps.append(amp / norm)
    amp_norms.append(norm)

    raw_phases.append(phase)
    raw_freqs.append(freqs_masked)
    valid_params.append(params)

print(f"Generated {len(raw_amps)} valid waveforms.")

# -----------------------------------------------------------------------------
# ## Step II: Define Sparse Frequency Grids and Interpolate
# -----------------------------------------------------------------------------
print("Step II: Creating sparse grids and interpolating...")

# Higher power corresponds to finer spacing at low frequencies and 
# low power corresponds to finer spacing at high frequencies.
sparse_freq_amp = generate_sparse_grid(f_min_grid, f_max_grid, num_points=200, power=1) 
sparse_freq_phase = generate_sparse_grid(f_min_grid, f_max_grid, num_points=200, power=4/3)

A_mat = np.zeros((len(sparse_freq_amp), len(raw_amps)))
Phi_mat = np.zeros((len(sparse_freq_phase), len(raw_phases)))

for i, (amp, phase, freqs) in enumerate(zip(raw_amps, raw_phases, raw_freqs)):
    uniq_idx = np.where(np.diff(freqs, prepend=freqs[0]-1e-12) > 0)[0]
    if uniq_idx.size < 2:
        raise RuntimeError(f"Too few unique frequency bins for sample {i}. Got {freqs.size} freq points.")
    freqs_unique = freqs[uniq_idx]
    amp_unique = amp[uniq_idx]
    phase_unique = phase[uniq_idx]

    k_amp = min(3, max(1, freqs.size - 1))
    k_phase = min(3, max(1, freqs.size - 1))

    spline_amp = UnivariateSpline(freqs, amp, s=0, k=k_amp, ext=2)
    spline_phase = UnivariateSpline(freqs, phase, s=0, k=k_phase, ext=2)

    fmin, fmax = freqs[0], freqs[-1]

    A_mat[:, i] = spline_amp(sparse_freq_amp)
    Phi_mat[:, i] = spline_phase(sparse_freq_phase)

    # inside_mask = (sparse_freq_amp >= fmin) & (sparse_freq_amp <= fmax)
    # outside_mask = ~inside_mask
    # if inside_mask.any():
    #     A_mat[inside_mask, i] = spline_amp(sparse_freq_amp[inside_mask])
    # if outside_mask.any():
        
    #     A_mat[outside_mask, i] = spline_amp(sparse_freq_amp[outside_mask])

    # inside_mask_p = (sparse_freq_phase >= fmin) & (sparse_freq_phase <= fmax)
    # outside_mask_p = ~inside_mask_p
    # if inside_mask_p.any():
    #     Phi_mat[inside_mask_p, i] = spline_phase(sparse_freq_phase[inside_mask_p])
    # if outside_mask_p.any():
    #     Phi_mat[outside_mask_p, i] = spline_phase(sparse_freq_phase[outside_mask_p])

# -----------------------------------------------------------------------------
# ## Step III: Compute Reduced Bases via SVD
# -----------------------------------------------------------------------------
print("Step III: Performing SVD to find reduced bases...")

Ua, sa, Vta = np.linalg.svd(A_mat, full_matrices=False)
Up, sp, Vtp = np.linalg.svd(Phi_mat, full_matrices=False)

# -----------------------------------------------------------------------------
# ## Plotting the SVD values
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

def plot_normalized_singular_values(sa, sp):
    """
    Plots normalized singular values and their cumulative sums
    for the amplitude and phase matrices.
    This helps in determining the effective rank of the matrices
    and selecting the truncation rank.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Singular Value Analysis", fontsize=16)

    # --- Amplitude singular values ---
    normalized_sa = sa / sa[0]
    x_sa = np.arange(1, len(sa) + 1)  # start x-axis at 1
    axs[0, 0].semilogy(x_sa, normalized_sa, '-o', markersize=4)
    axs[0, 0].set_ylim(min(normalized_sa), 1.0)
    axs[0, 0].set_title("Amplitude Singular Values", fontsize=14)
    axs[0, 0].set_xlabel("Singular Value Index", fontsize=12)
    axs[0, 0].set_ylabel("Normalized Singular Value", fontsize=12)
    axs[0, 0].grid(True, which="both", ls="--")

    # Cumulative sum (Amplitude)
    cumsum_sa = np.cumsum(sa) / np.sum(sa)
    axs[1, 0].plot(x_sa, cumsum_sa, '-o', markersize=4)
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].set_title("Cumulative Sum (Amplitude)", fontsize=14)
    axs[1, 0].set_xlabel("Singular Value Index", fontsize=12)
    axs[1, 0].set_ylabel("Cumulative Energy", fontsize=12)
    axs[1, 0].grid(True, which="both", ls="--")

    # --- Phase singular values ---
    normalized_sp = sp / sp[0]
    x_sp = np.arange(1, len(sp) + 1)
    axs[0, 1].semilogy(x_sp, normalized_sp, '-o', markersize=4, color='red')
    axs[0, 1].set_ylim(min(normalized_sp), 1.0)
    axs[0, 1].set_title("Phase Singular Values", fontsize=14)
    axs[0, 1].set_xlabel("Singular Value Index", fontsize=12)
    axs[0, 1].set_ylabel("Normalized Singular Value", fontsize=12)
    axs[0, 1].grid(True, which="both", ls="--")

    # Cumulative sum (Phase)
    cumsum_sp = np.cumsum(sp) / np.sum(sp)
    axs[1, 1].plot(x_sp, cumsum_sp, '-o', markersize=4, color='red')
    axs[1, 1].set_ylim(0, 1.05)
    axs[1, 1].set_title("Cumulative Sum (Phase)", fontsize=14)
    axs[1, 1].set_xlabel("Singular Value Index", fontsize=12)
    axs[1, 1].set_ylabel("Cumulative Energy", fontsize=12)
    axs[1, 1].grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{args.results_dir}/singular_value_analysis.pdf", dpi=300)
    plt.show()


plot_normalized_singular_values(sa, sp)

rank_a = 200
rank_p = 150

B_a = Ua[:, :rank_a] 
B_p = Up[:, :rank_p] 

# -----------------------------------------------------------------------------
# ## Step IV: Interpolate Projection Coefficients
# -----------------------------------------------------------------------------
print("Step IV: Interpolating projection coefficients...")

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
# ## Check smoothness of projection coefficients
# -----------------------------------------------------------------------------
print("Checking smoothness of projection coefficients...")

modes_to_plot = [0, 1, 2, 3, 10, 20]

fig = plt.figure(figsize=(14, 4*len(modes_to_plot)))
fig.suptitle("Projection Coefficients across Parameter Space", fontsize=16)

for j, mode in enumerate(modes_to_plot):

    coeff_grid_a = Ca[mode, :].reshape(len(chi_unique), len(q_unique))
    coeff_grid_p = Cp[mode, :].reshape(len(chi_unique), len(q_unique))

    Q, Chi = np.meshgrid(q_unique, chi_unique)

    ax1 = fig.add_subplot(len(modes_to_plot), 2, 2*j+1, projection='3d')
    surf_a = ax1.plot_surface(Q, Chi, coeff_grid_a, cmap="viridis", edgecolor="none")
    ax1.set_title(f"Amplitude Coefficient {mode}")
    ax1.set_xlabel("Mass ratio q")
    ax1.set_ylabel(rf"Spin $\chi$")
    ax1.set_zlabel("Coefficient")
    ax1.zaxis.labelpad = 15
    fig.colorbar(surf_a, ax=ax1, shrink=0.6, aspect=10, pad=0.2) 

    ax2 = fig.add_subplot(len(modes_to_plot), 2, 2*j+2, projection='3d')
    surf_p = ax2.plot_surface(Q, Chi, coeff_grid_p, cmap="plasma", edgecolor="none")
    ax2.set_title(f"Phase Coefficient {mode}")
    ax2.set_xlabel("Mass ratio q")
    ax2.set_ylabel(rf"Spin $\chi$")
    ax2.set_zlabel("Coefficient")
    ax2.zaxis.labelpad = 15
    fig.colorbar(surf_p, ax=ax2, shrink=0.6, aspect=10, pad=0.2) 

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.3)
plt.savefig(f"{args.results_dir}/projection_coefficients.pdf", dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# ## Plotting the normalization factor variation
# -----------------------------------------------------------------------------
def plot_normalization_factor(amp_norms_grid, q_unique, chi_unique):
    """
    Plots the variation of the normalization factor across the parameter space.

    Parameters:
    -----------
    amp_norms_grid : 2D numpy array
        A grid of the normalization factors.
    q_unique : 1D numpy array
        The unique values for the mass ratio q.
    chi_unique : 1D numpy array
        The unique values for the spin parameter chi.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a meshgrid for plotting
    Q, Chi = np.meshgrid(q_unique, chi_unique)

    # Create the surface plot
    surf = ax.plot_surface(Q, Chi, amp_norms_grid, cmap="viridis", edgecolor="none")
    ax.set_title("Variation of Waveform Normalization Factor", fontsize=16)
    ax.set_xlabel("Mass ratio q", fontsize=12)
    ax.set_ylabel(r"Spin $\chi$", fontsize=12)
    ax.set_zlabel("Normalization Factor (Norm)", fontsize=12)
    ax.zaxis.labelpad = 15

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/normalization_factor_variation.pdf", dpi=300)
    plt.show()

print("Plotting the variation of the normalization factor...")
plot_normalization_factor(amp_norms_grid, q_unique, chi_unique)

# -----------------------------------------------------------------------------
# ## Step V: Assemble and Evaluate the Surrogate Model
# -----------------------------------------------------------------------------
print("Step V: Assembling the surrogate model evaluator.")

def evaluate_surrogate_fd(q_star, chi_star, freqs_out):
    """
    Evaluates the surrogate model at a new parameter point (q*, chi*).
    This function implements Eq. (6.10) from the paper. 
    """
    ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_a])
    cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_p])

    amp_recon_sparse = B_a @ ca_star
    phase_recon_sparse = B_p @ cp_star

    spline_amp = UnivariateSpline(sparse_freq_amp, amp_recon_sparse, s=0, k=min(3, max(1, sparse_freq_amp.size-1)), ext=2)
    spline_phase = UnivariateSpline(sparse_freq_phase, phase_recon_sparse, s=0, k=min(3, max(1, sparse_freq_phase.size-1)), ext=2)
    
    amp_final = spline_amp(freqs_out)
    phase_final = spline_phase(freqs_out)

    norm_star = interp_amp_norm(chi_star, q_star)[0, 0]
    amp_final *= norm_star

    h_fd_recon = amp_final * np.exp(1j * phase_final)
    
    return freqs_out, h_fd_recon

print("\nValidating model with a test waveform...")

# -----------------------------------------------------------------------------
# ## Run a test case to validate the surrogate model
# -----------------------------------------------------------------------------
test_params = {'q': 8.23, 'chi': -0.5}
# test_params = {'q': 4.5, 'chi': 0.45}
# test_params = {'q': 1.23, 'chi': -0.7}

true_freqs, true_h_fd = generate_fd_waveform(test_params, f_lower, delta_t, window_type=window_type, epsilon=0.2)
mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
true_freqs_masked = true_freqs[mask]
true_h_fd_masked = true_h_fd[mask]
true_amp, true_phase = get_amp_phase(true_freqs_masked, true_h_fd_masked)


surr_freqs, surr_h_fd = evaluate_surrogate_fd(test_params['q'], test_params['chi'], true_freqs_masked)

# plt.plot(true_freqs_masked, np.abs(true_h_fd_masked), label='True Waveform', lw=2)
# plt.plot(surr_freqs, np.abs(surr_h_fd), '--', label='Surrogate Model', lw=2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title(f"Surrogate Model Validation for q={test_params['q']}, $\chi$={test_params['chi']}")
# plt.legend()
# plt.grid(True, which="both", ls="--")

surr_amp = np.abs(surr_h_fd)
surr_phase = np.unwrap(np.angle(surr_h_fd))

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(rf"Surrogate Model Validation for q={test_params['q']}, $\chi$={test_params['chi']}", fontsize=16)

axs[0].loglog(true_freqs_masked, true_amp, label='True Waveform', lw=3, alpha=0.8)
axs[0].loglog(surr_freqs, surr_amp, '--', label='Surrogate Model', lw=2, color='red')
axs[0].set_ylabel('Amplitude', fontsize=12)
axs[0].legend(fontsize=11)
axs[0].set_title('Amplitude Comparison', fontsize=14)
axs[0].grid(True, which="both", ls="--")

axs[1].semilogx(true_freqs_masked, true_phase, label='True Waveform (centered)', lw=3, alpha=0.8)
axs[1].semilogx(surr_freqs, surr_phase, '--', label='Surrogate Model (centered)', lw=2, color='red')
axs[1].set_xlabel('Frequency (Hz)', fontsize=12)
axs[1].set_ylabel('Phase (rad)', fontsize=12)
axs[1].legend(fontsize=11)
axs[1].set_title('Phase Comparison', fontsize=14)
axs[1].grid(True, which="both", ls="--")

# -----------------------------------------------------------------------------
# ## Compute mismatch
# -----------------------------------------------------------------------------
pycbc_surr_h_fd = pycbc.types.FrequencySeries(surr_h_fd, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
pycbc_true_h_fd = pycbc.types.FrequencySeries(true_h_fd_masked, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
pycbc_surr_h_fd.start_time = 0
pycbc_true_h_fd.start_time = 0

mismatch = 1 - pycbc.filter.matchedfilter.optimized_match(pycbc_surr_h_fd, pycbc_true_h_fd, psd=pycbc.psd.aLIGOZeroDetHighPower(len(pycbc_true_h_fd), pycbc_true_h_fd.delta_f, f_lower), low_frequency_cutoff=f_min_grid)[0]
print(f"Mismatch between surrogate model and true model = {mismatch:.3e}")

# -----------------------------------------------------------------------------
# ## Plot the results
# -----------------------------------------------------------------------------
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{args.results_dir}/Surrogate_Model_vs_True_Model_1.pdf')
plt.show()

# -----------------------------------------------------------------------------
# ## Save the Surrogate Model
# -----------------------------------------------------------------------------
def save_surrogate(filename, data):
    """Save surrogate model data to disk."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)

surrogate_data = {
    "sparse_freq_amp": sparse_freq_amp,
    "sparse_freq_phase": sparse_freq_phase,
    "B_a": B_a,
    "B_p": B_p,
    "Ca": Ca,
    "Cp": Cp,
    "amp_norms_grid": amp_norms_grid,
    "q_unique": q_unique,
    "chi_unique": chi_unique
}

save_surrogate("Models/surrogate_model.pkl", surrogate_data)
print("Surrogate model saved to surrogate_model.pkl")
