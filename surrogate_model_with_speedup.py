import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from pycbc.waveform import get_td_waveform
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, griddata

# -----------------------------------------------------------------------------
# ## Setup and Helper Functions
# -----------------------------------------------------------------------------
def generate_fd_waveform(params, M_total, f_lower, delta_t, nfft):
    """Generates a time-domain waveform, applies a window, and converts to frequency domain."""
    q = params['q']
    chi = params['chi']
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

    tukey_window = windows.tukey(len(h_td_raw), alpha=0.1)
    h_td_windowed = h_td_raw * tukey_window

    if len(h_td_windowed) < nfft:
        h_td = np.pad(h_td_windowed, (0, nfft - len(h_td_windowed)))
    else:
        h_td = h_td_windowed[:nfft]

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
    Implements a simplified version of the 'Constant Spline Error' (CSE) grid generation.
    The grid spacing Delta(f) is proportional to f^power.
    """
    if num_points <= 1:
        return np.array([f_min])
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

q_vals = np.linspace(1, 10, 10)
chi_vals = np.linspace(-0.8, 0.6, 30)
param_grid_q, param_grid_chi = np.meshgrid(q_vals, chi_vals)
params_list = [{'q': q, 'chi': chi} for q, chi in zip(param_grid_q.flatten(), param_grid_chi.flatten())]

f_lower = 20.0
f_min_grid = 25.0
f_max_grid = 1024.0
delta_t = 1/4096
nfft = 16 * 4096

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
        freqs, h_fd = generate_fd_waveform(params, M_total, f_lower, delta_t, nfft)
        if freqs is None:
            continue

        mask = (freqs >= f_min_grid) & (freqs <= f_max_grid)
        freqs_masked = freqs[mask]

        if freqs_masked.size == 0:
            continue

        amp, phase = get_amp_phase(freqs_masked, h_fd[mask])

        if amp.size == 0:
            continue

        norm = np.linalg.norm(amp)
        if norm == 0 or not np.isfinite(norm):
            continue

        raw_amps.append(amp / norm)
        amp_norms.append(norm)
        raw_phases.append(phase)
        raw_freqs.append(freqs_masked)
        valid_params.append(params)

    print(f"Generated {len(raw_amps)} valid waveforms for M_total={M_total}.")

    if len(raw_amps) == 0:
        print("No valid waveforms for this M_total; skipping.")
        true_times.append(np.nan)
        surr_times.append(np.nan)
        speedups.append(np.nan)
        continue

    # -----------------------------------------------------------------------------
    # ## Step II: Define Sparse Frequency Grids and Interpolate
    # -----------------------------------------------------------------------------
    print("Step II: Creating sparse grids and interpolating...")

    sparse_freq_amp = generate_sparse_grid(f_min_grid, f_max_grid, num_points=200, power=0.3)
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

        k_amp = min(3, max(1, freqs_unique.size - 1))
        k_phase = min(3, max(1, freqs_unique.size - 1))

        spline_amp = UnivariateSpline(freqs_unique, amp_unique, s=0, k=k_amp, ext=3)
        spline_phase = UnivariateSpline(freqs_unique, phase_unique, s=0, k=k_phase, ext=3)

        fmin, fmax = freqs_unique[0], freqs_unique[-1]

        inside_mask = (sparse_freq_amp >= fmin) & (sparse_freq_amp <= fmax)
        A_mat[inside_mask, i] = spline_amp(sparse_freq_amp[inside_mask])
        outside_mask = ~inside_mask
        if outside_mask.any():
            A_mat[outside_mask, i] = spline_amp(sparse_freq_amp[outside_mask])

        inside_mask_p = (sparse_freq_phase >= fmin) & (sparse_freq_phase <= fmax)
        Phi_mat[inside_mask_p, i] = spline_phase(sparse_freq_phase[inside_mask_p])
        outside_mask_p = ~inside_mask_p
        if outside_mask_p.any():
            Phi_mat[outside_mask_p, i] = spline_phase(sparse_freq_phase[outside_mask_p])

    # -----------------------------------------------------------------------------
    # ## Step III: Compute Reduced Bases via SVD
    # -----------------------------------------------------------------------------
    print("Step III: Performing SVD to find reduced bases...")
    Ua, sa, Vta = np.linalg.svd(A_mat, full_matrices=False)
    Up, sp, Vtp = np.linalg.svd(Phi_mat, full_matrices=False)


    rank_a = min(30, Ua.shape[1])
    rank_p = min(60, Up.shape[1])
    B_a = Ua[:, :rank_a]
    B_p = Up[:, :rank_p]

    # -----------------------------------------------------------------------------
    # ## Step IV: Interpolate Projection Coefficients
    # -----------------------------------------------------------------------------
    print("Step IV: Interpolating projection coefficients...")

    Ca = B_a.T @ A_mat
    Cp = B_p.T @ Phi_mat

    q_unique = q_vals.copy()
    chi_unique = chi_vals.copy()

    points = np.array([[p['chi'], p['q']] for p in valid_params])
    grid_chi, grid_q = np.meshgrid(chi_unique, q_unique, indexing='ij')
    grid_points = np.vstack([grid_chi.ravel(), grid_q.ravel()]).T

    interpolants_a = []
    for i in range(rank_a):
        values = Ca[i, :]
        
        grid_vals = griddata(points, values, grid_points, method='linear')

        if np.any(np.isnan(grid_vals)):
            grid_vals_nearest = griddata(points, values, grid_points, method='nearest')
            nan_mask = np.isnan(grid_vals)
            grid_vals[nan_mask] = grid_vals_nearest[nan_mask]
        coeff_grid = grid_vals.reshape(len(chi_unique), len(q_unique))
        interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
        interpolants_a.append(interp)

    interpolants_p = []
    for i in range(rank_p):
        values = Cp[i, :]
        grid_vals = griddata(points, values, grid_points, method='linear')
        if np.any(np.isnan(grid_vals)):
            grid_vals_nearest = griddata(points, values, grid_points, method='nearest')
            nan_mask = np.isnan(grid_vals)
            grid_vals[nan_mask] = grid_vals_nearest[nan_mask]
        coeff_grid = grid_vals.reshape(len(chi_unique), len(q_unique))
        interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
        interpolants_p.append(interp)

    amp_norms_arr = np.array(amp_norms)
    
    grid_vals = griddata(points, amp_norms_arr, grid_points, method='linear')
    if np.any(np.isnan(grid_vals)):
        grid_vals_nearest = griddata(points, amp_norms_arr, grid_points, method='nearest')
        nan_mask = np.isnan(grid_vals)
        grid_vals[nan_mask] = grid_vals_nearest[nan_mask]
    amp_norms_grid = grid_vals.reshape(len(chi_unique), len(q_unique))
    interp_amp_norm = RectBivariateSpline(chi_unique, q_unique, amp_norms_grid, kx=3, ky=3)

    # -----------------------------------------------------------------------------
    # ## Step V: Assemble and Evaluate the Surrogate Model
    # -----------------------------------------------------------------------------
    print("Step V: Assembling the surrogate model evaluator.")

    def evaluate_surrogate_fd(q_star, chi_star, freqs_out):
        """
        Evaluates the surrogate model at a new parameter point (q*, chi*).
        """
        ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_a])
        cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in interpolants_p])

        amp_recon_sparse = B_a @ ca_star
        phase_recon_sparse = B_p @ cp_star

        spline_amp = UnivariateSpline(sparse_freq_amp, amp_recon_sparse, s=0,
                                     k=min(3, max(1, sparse_freq_amp.size-1)), ext=3)
        spline_phase = UnivariateSpline(sparse_freq_phase, phase_recon_sparse, s=0,
                                       k=min(3, max(1, sparse_freq_phase.size-1)), ext=3)

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
        true_freqs, true_h_fd = generate_fd_waveform(test_params, M_total, f_lower, delta_t, nfft)
    end = time.time()
    true_time = (end - start) / n_trials
    true_times.append(true_time)

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

    speedup = true_time / surr_time if (surr_time != 0 and np.isfinite(surr_time)) else np.nan
    speedups.append(speedup)

# ----------------------------------------------------
# Plot absolute runtimes
# ----------------------------------------------------
os.makedirs("Results", exist_ok=True)

plt.figure(figsize=(8,6))
plt.plot(M_total_list, true_times, marker="o", lw=2, label="True Model")
plt.plot(M_total_list, surr_times, marker="o", lw=2, label="Surrogate Model")
plt.xlabel("Total Mass $M_{tot} \\, (M_\\odot)$", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.title(f"Runtime Comparison (q={q_fixed}, χ={chi_fixed})", fontsize=14)
plt.legend()
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.savefig("Results/Runtime_vs_Mass.pdf")
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
plt.savefig("Results/Speedup_vs_Mass.pdf")
plt.show()
