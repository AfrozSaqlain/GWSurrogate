import os
import pycbc
import pickle
import argparse
import multiprocessing

import pycbc.psd

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from modules.constants import *
from modules.gw_utils import generate_fd_waveform, get_amp_phase, generate_sparse_grid

parser = argparse.ArgumentParser(description="Train a surrogate model for gravitational waveforms.")
parser.add_argument('--results-dir', type=str, default='Results', help='Directory to save results and plots.')
args = parser.parse_args()

if args.results_dir != 'Results' and not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# -----------------------------------------------------------------------------
# ## Step I: Generate and Pre-process Training Waveforms
# -----------------------------------------------------------------------------
print("Step I: Generating training data...")

def process_waveform(params, f_lower, delta_t, window_type, LAL_taper_method, epsilon, num_extrema_start, num_extrema_end, f_min_mask, f_max_mask):
    
    freqs, h_fd = generate_fd_waveform(params=params, f_lower=f_lower, delta_t=delta_t, window_type=window_type, LAL_taper_method=LAL_taper_method, padding_type='power_of_2', epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end)
    mask = (freqs >= f_min_mask) & (freqs <= f_max_mask)
    freqs_masked = freqs[mask]

    amp, phase = get_amp_phase(freqs_masked, h_fd[mask])
    df = freqs_masked[1] - freqs_masked[0]
    norm = np.sqrt(np.sum(amp**2) * df)
    
    return (amp / norm, norm, phase, freqs_masked, params)

bounds_q = (1, 10)
bounds_chi = (-1, 1)

q_vals = np.concatenate((
    np.linspace(bounds_q[0], bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.2, 30, endpoint=False),
    np.linspace(bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.2, bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.8, 10, endpoint=False),
    np.linspace(bounds_q[0] + (bounds_q[1] - bounds_q[0]) * 0.8, bounds_q[1], 20)
))

chi_vals = np.concatenate((
    np.linspace(bounds_chi[0], bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.2, 30, endpoint=False),
    np.linspace(bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.2, bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.8, 10, endpoint=False),
    np.linspace(bounds_chi[0] + (bounds_chi[1] - bounds_chi[0]) * 0.8, bounds_chi[1], 20)
))

param_grid_q, param_grid_chi = np.meshgrid(q_vals, chi_vals)
params_list = [{'q': q, 'chi': chi} for q, chi in zip(param_grid_q.flatten(), param_grid_chi.flatten())]

args_for_starmap = [(p, f_lower, delta_t, window_type, LAL_taper_method, epsilon, num_extrema_start, num_extrema_end, f_min_mask, f_max_mask) for p in params_list]

num_processes = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=num_processes) as pool:
    results = pool.starmap(process_waveform, args_for_starmap)

valid_results = [r for r in results if r is not None]
raw_amps, amp_norms, raw_phases, raw_freqs, valid_params = zip(*valid_results)

print(f"Generated {len(raw_amps)} valid waveforms.")

# -----------------------------------------------------------------------------
# ## Step II: Define Sparse Frequency Grids and Interpolate
# -----------------------------------------------------------------------------
print("Step II: Creating sparse grids and interpolating...")

f_min_grid = f_min_mask
f_max_grid = f_max_mask

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
print("Step III: Performing SVD to find reduced bases...")

Ua, sa, Vta = np.linalg.svd(A_mat, full_matrices=False)
Up, sp, Vtp = np.linalg.svd(Phi_mat, full_matrices=False)

rank_a = 10
rank_p = 10

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
# ## Step V: Assemble and Evaluate the Surrogate Model
# -----------------------------------------------------------------------------
print("Step V: Assembling the surrogate model evaluator.")

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

print("\nValidating model with a test waveform...")

# -----------------------------------------------------------------------------
# ## Run a test case to validate the surrogate model
# -----------------------------------------------------------------------------
test_params = {'q': 8.23, 'chi': -0.5}
# test_params = {'q': 4.5, 'chi': 0.45}
# test_params = {'q': 1.23, 'chi': -0.7}

true_freqs, true_h_fd = generate_fd_waveform(test_params, f_lower, delta_t, window_type=window_type, padding_type='power_of_2', epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end)
mask = (true_freqs >= f_min_grid) & (true_freqs <= f_max_grid)
true_freqs_masked = true_freqs[mask]
true_h_fd_masked = true_h_fd[mask]
true_amp, true_phase = get_amp_phase(true_freqs_masked, true_h_fd_masked)

surr_freqs, surr_h_fd = evaluate_surrogate_fd(test_params['q'], test_params['chi'], true_freqs_masked)
surr_amp, surr_phase = get_amp_phase(surr_freqs, surr_h_fd)

# -----------------------------------------------------------------------------
# ## Compute mismatch
# -----------------------------------------------------------------------------
pycbc_surr_h_fd = pycbc.types.FrequencySeries(surr_h_fd, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
pycbc_true_h_fd = pycbc.types.FrequencySeries(true_h_fd_masked, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
pycbc_surr_h_fd.start_time = 0
pycbc_true_h_fd.start_time = 0

mismatch = 1 - pycbc.filter.matchedfilter.optimized_match(pycbc_surr_h_fd, pycbc_true_h_fd, psd=pycbc.psd.aLIGOZeroDetHighPower(len(pycbc_true_h_fd), pycbc_true_h_fd.delta_f, f_lower), low_frequency_cutoff=f_min_grid)[0]
print(f"Mismatch between surrogate model and true model = {mismatch:.3e}\n")

# -----------------------------------------------------------------------------
# ## Plot the results
# -----------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle(rf"Surrogate Model Validation for q={test_params['q']}, $\chi$={test_params['chi']}", fontsize=16)

axs[0].loglog(true_freqs_masked, true_amp, label='True Waveform', lw=3, alpha=0.8)
axs[0].loglog(surr_freqs, surr_amp, '--', label='Surrogate Model', lw=2, color='red')
axs[0].set_ylabel('Amplitude', fontsize=12)
axs[0].legend(fontsize=11)
axs[0].set_title('Amplitude Comparison', fontsize=14)
axs[0].grid(True, which="both", ls="--")

axs[1].semilogx(true_freqs_masked, true_phase, label='True Waveform', lw=3, alpha=0.8)
axs[1].semilogx(surr_freqs, surr_phase, '--', label='Surrogate Model', lw=2, color='red')
axs[1].set_xlabel('Frequency (Hz)', fontsize=12)
axs[1].set_ylabel('Phase (rad)', fontsize=12)
axs[1].legend(fontsize=11)
axs[1].set_title('Phase Comparison', fontsize=14)
axs[1].grid(True, which="both", ls="--")

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
print("Surrogate model saved to surrogate_model.pkl\n")















# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ########################## Some additional plots ############################
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ## Plotting the SVD values
# -----------------------------------------------------------------------------
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
    x_sa = np.arange(1, len(sa) + 1)
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
    plt.savefig(f"{args.results_dir}/Singular_value_analysis.pdf", dpi=300)
    plt.show()

print("Analyzing SVD singular values' fall off...")
plot_normalized_singular_values(sa, sp)

# -----------------------------------------------------------------------------
# ## Check smoothness of projection coefficients
# -----------------------------------------------------------------------------
print("Checking smoothness of projection coefficients...")

modes_to_plot = [0, 1, 2, 3, 4, 9]

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
plt.savefig(f"{args.results_dir}/Projection_coefficients.pdf", dpi=300)
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

    Q, Chi = np.meshgrid(q_unique, chi_unique)

    surf = ax.plot_surface(Q, Chi, amp_norms_grid, cmap="viridis", edgecolor="none")
    ax.set_title("Variation of Waveform Normalization Factor", fontsize=16)
    ax.set_xlabel("Mass ratio q", fontsize=12)
    ax.set_ylabel(r"Spin $\chi$", fontsize=12)
    ax.set_zlabel("Normalization Factor (Norm)", fontsize=12)
    ax.zaxis.labelpad = 15

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)

    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/Normalization_factor_variation.pdf", dpi=300)
    plt.show()

print("Plotting the variation of the normalization factor...")
plot_normalization_factor(amp_norms_grid, q_unique, chi_unique)

# -----------------------------------------------------------------------------
# ## Plotting the Basis Functions
# -----------------------------------------------------------------------------
def plot_basis_functions(basis_matrix, freq_grid, modes_to_plot, basis_type):
    """
    Plots the specified basis functions against their frequency grid.

    This helps visualize the principal components of the waveform model.

    Parameters:
    -----------
    basis_matrix : 2D numpy array
        Matrix whose columns are the basis functions (e.g., B_a or B_p).
    freq_grid : 1D numpy array
        The sparse frequency grid corresponding to the basis.
    modes_to_plot : list of int
        A list of indices for the basis functions (modes) to plot.
    basis_type : str
        A string ('Amplitude' or 'Phase') to label the plot title and filename.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    for mode in modes_to_plot:
        if mode < basis_matrix.shape[1]:
            plt.plot(freq_grid, basis_matrix[:, mode], label=f'Basis Function {mode}', lw=1, marker='o', markersize=1.5)
        else:
            print(f"Warning: Mode {mode} is out of bounds for the given basis matrix.")

    plt.title(f'{basis_type} Basis Functions', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Basis Function Value', fontsize=12)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.results_dir}/{basis_type}_basis_functions.pdf', dpi=300)
    plt.show()

modes_to_plot = np.arange(0, 9)

print("Plotting basis functions for amplitude and phase...")
plot_basis_functions(B_a, sparse_freq_amp, modes_to_plot, 'Amplitude')
plot_basis_functions(B_p, sparse_freq_phase, modes_to_plot, 'Phase')

# -----------------------------------------------------------------------------
# ## Plotting the waveforms in frequency domain
# -----------------------------------------------------------------------------
# plt.plot(true_freqs_masked, np.abs(true_h_fd_masked), label='True Waveform', lw=2)
# plt.plot(surr_freqs, np.abs(surr_h_fd), '--', label='Surrogate Model', lw=2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title(f"Surrogate Model Validation for q={test_params['q']}, $\chi$={test_params['chi']}")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()