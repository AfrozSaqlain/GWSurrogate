import os
import pickle
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import pycbc
import pycbc.psd

from surrogw.modules.gw_utils import generate_fd_waveform
from surrogw.modules.constants import f_lower, delta_t, f_min_mask, f_max_mask, window_type, LAL_taper_method, epsilon, num_extrema_start, num_extrema_end

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Train a surrogate model for gravitational waveforms.")
parser.add_argument('--results-dir', type=str, default='Results', help='Directory to save results and plots.')
parser.add_argument('--nprocs', type=int, default=None, help='Number of worker processes to use. Defaults to cpu_count().')
parser.add_argument('--surrogate-file', type=str, default="Models/surrogate_model.pkl", help='Path to surrogate pickle.')
args = parser.parse_args()

if args.results_dir != 'Results' and not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

NPROCS = args.nprocs or os.cpu_count()

# ----------------------------
# Load surrogate raw data
# ----------------------------
def load_surrogate_raw(filename):
    """Load surrogate pickle and return raw arrays that are picklable. Do NOT build spline objects yet."""
    with open(filename, "rb") as f:
        data = pickle.load(f)

    raw = {
        "sparse_freq_amp": np.array(data["sparse_freq_amp"]),
        "sparse_freq_phase": np.array(data["sparse_freq_phase"]),
        "B_a": np.array(data["B_a"]),
        "B_p": np.array(data["B_p"]),
        "Ca": np.array(data["Ca"]),
        "Cp": np.array(data["Cp"]),
        "chi_unique": np.array(data["chi_unique"]),
        "q_unique": np.array(data["q_unique"]),
        "amp_norms_grid": np.array(data["amp_norms_grid"])
    }
    return raw

surrogate_raw = load_surrogate_raw(args.surrogate_file)
print("Surrogate raw data loaded.")

# ----------------------------
# Worker-side global cache (will be set by initializer)
# ----------------------------
# On worker processes, these globals will be populated
SURR_RAW = None
SURR_CACHE = None
F_MIN_GRID = None
F_MAX_GRID = None
F_LOWER = None
DELTA_T = None
EPSILON = None
WINDOW_TYPE = None
LAL_TAPER_METHOD = None

def init_worker(surr_raw, f_min_grid, f_max_grid, f_lower, delta_t, window_type, epsilon, lal_taper_method, num_extrema_start, num_extrema_end):
    """Initializer for worker processes: sets global raw surrogate and constants."""
    global SURR_RAW, SURR_CACHE, F_MIN_GRID, F_MAX_GRID, F_LOWER, DELTA_T, EPSILON, WINDOW_TYPE, LAL_TAPER_METHOD, NUM_EXTREMA_START, NUM_EXTREMA_END

    SURR_RAW = surr_raw
    SURR_CACHE = {
        "interpolants_a": None,
        "interpolants_p": None,
        "interp_amp_norm": None
    }
    F_MIN_GRID = f_min_grid
    F_MAX_GRID = f_max_grid
    F_LOWER = f_lower
    DELTA_T = delta_t
    EPSILON = epsilon
    WINDOW_TYPE= window_type
    LAL_TAPER_METHOD = lal_taper_method
    NUM_EXTREMA_START = num_extrema_start
    NUM_EXTREMA_END = num_extrema_end

def build_interpolants_if_needed():
    """Build RectBivariateSpline interpolants in the worker process on first use."""
    global SURR_RAW, SURR_CACHE
    if SURR_CACHE["interpolants_a"] is None:
        chi_u = SURR_RAW["chi_unique"]
        q_u = SURR_RAW["q_unique"]
        # Build amplitude interpolants
        interps_a = []
        for i in range(SURR_RAW["Ca"].shape[0]):
            coeff_grid = SURR_RAW["Ca"][i, :].reshape(len(chi_u), len(q_u))
            interps_a.append(RectBivariateSpline(chi_u, q_u, coeff_grid, kx=3, ky=3))
        # Build phase interpolants
        interps_p = []
        for i in range(SURR_RAW["Cp"].shape[0]):
            coeff_grid = SURR_RAW["Cp"][i, :].reshape(len(chi_u), len(q_u))
            interps_p.append(RectBivariateSpline(chi_u, q_u, coeff_grid, kx=3, ky=3))
        interp_amp_norm = RectBivariateSpline(chi_u, q_u, SURR_RAW["amp_norms_grid"], kx=3, ky=3)
        SURR_CACHE["interpolants_a"] = interps_a
        SURR_CACHE["interpolants_p"] = interps_p
        SURR_CACHE["interp_amp_norm"] = interp_amp_norm

def evaluate_surrogate_fd_worker(q_star, chi_star, freqs_out):
    """Evaluate surrogate using worker-side cached interpolants."""
    build_interpolants_if_needed()
    ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in SURR_CACHE["interpolants_a"]])
    cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in SURR_CACHE["interpolants_p"]])

    amp_recon_sparse = SURR_RAW["B_a"] @ ca_star
    phase_recon_sparse = SURR_RAW["B_p"] @ cp_star

    spline_amp = UnivariateSpline(SURR_RAW["sparse_freq_amp"], amp_recon_sparse, s=0, k=3, ext=2)
    spline_phase = UnivariateSpline(SURR_RAW["sparse_freq_phase"], phase_recon_sparse, s=0, k=3, ext=2)

    amp_final = spline_amp(freqs_out)
    phase_final = spline_phase(freqs_out)

    norm_star = SURR_CACHE["interp_amp_norm"](chi_star, q_star)[0, 0]
    amp_final *= norm_star

    return freqs_out, amp_final * np.exp(1j * phase_final)

def compute_mismatch_point(q, chi):
    """Worker function: generate waveform, evaluate surrogate at masked freqs, compute mismatch."""
    # generate waveform
    params = {"q": q, "chi": chi}

    freqs, h_fd = generate_fd_waveform(params, F_LOWER, DELTA_T, window_type=WINDOW_TYPE, LAL_taper_method=LAL_taper_method, padding_type='power_of_2', epsilon=EPSILON, num_extrema_start=NUM_EXTREMA_START, num_extrema_end=NUM_EXTREMA_END)

    mask = (freqs >= F_MIN_GRID) & (freqs <= F_MAX_GRID)
    
    freqs_masked = freqs[mask]
    h_fd_masked = h_fd[mask]

    _, surr_h_fd = evaluate_surrogate_fd_worker(q, chi, freqs_masked)

    delta_f = freqs_masked[1] - freqs_masked[0]

    pycbc_surr_h_fd = pycbc.types.FrequencySeries(surr_h_fd, delta_f=delta_f, epoch=0)
    pycbc_true_h_fd = pycbc.types.FrequencySeries(h_fd_masked, delta_f=delta_f, epoch=0)

    pycbc_surr_h_fd.start_time = 0
    pycbc_true_h_fd.start_time = 0

    psd = pycbc.psd.aLIGOZeroDetHighPower(len(pycbc_true_h_fd), pycbc_true_h_fd.delta_f, F_LOWER)

    match_tuple = pycbc.filter.matchedfilter.optimized_match(
        pycbc_surr_h_fd, pycbc_true_h_fd,
        psd=psd,
        low_frequency_cutoff=F_MIN_GRID
    )
    
    match_val = match_tuple[0]
    mismatch = 1.0 - match_val
    return mismatch


# ----------------------------
# Main: grid, parallel execution
# ----------------------------
f_min_grid = f_min_mask
f_max_grid = f_max_mask

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

pairs = [(q, chi) for chi in chi_vals for q in q_vals] 


print(f"Starting parallel mismatch computation with {NPROCS} processes...")
mismatch_flat = np.full(len(pairs), np.nan)

with ProcessPoolExecutor(max_workers=NPROCS,
                         initializer=init_worker,
                         initargs=(surrogate_raw, f_min_grid, f_max_grid, f_lower, delta_t, window_type, epsilon, LAL_taper_method, num_extrema_start, num_extrema_end)) as exe:
    futures = {exe.submit(compute_mismatch_point, q, chi): i for i, (q, chi) in enumerate(pairs)}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        idx = futures[fut]
        mismatch_flat[idx] = fut.result()

# reshape result to (len(chi_vals), len(q_vals))
mismatch_vals = mismatch_flat.reshape(len(chi_vals), len(q_vals))

# ----------------------------
# Interpolation & Plotting
# ----------------------------
Q, Chi = np.meshgrid(q_vals, chi_vals)

interp_mismatch = RegularGridInterpolator(
    (chi_vals, q_vals), mismatch_vals,
    method="linear", bounds_error=False, fill_value=np.nan
)

q_fine = np.linspace(q_vals.min(), q_vals.max(), 400)
chi_fine = np.linspace(chi_vals.min(), chi_vals.max(), 400)
Q_fine, Chi_fine = np.meshgrid(q_fine, chi_fine)

mismatch_fine = interp_mismatch((Chi_fine, Q_fine))

# For LogNorm we need vmin>0; use nanmin/nanmax and ensure minimal positive eps
eps = 1e-12
vmin = np.nanmin(mismatch_vals)
vmax = np.nanmax(mismatch_vals)
if math.isnan(vmin) or math.isnan(vmax):
    vmin, vmax = eps, 1.0
else:
    # ensure vmin positive
    vmin = max(eps, vmin)
    vmax = max(vmin * 10.0, vmax)  # ensure vmax > vmin

plt.pcolormesh(
    Q, Chi, mismatch_vals,
    shading='auto',
    cmap='viridis',
    norm=colors.LogNorm(vmin=vmin, vmax=vmax)
)
plt.colorbar(label='Mismatch (log scale)')
plt.xlabel('Mass Ratio q')
plt.ylabel('Spin χ')
plt.title('Mismatch between Surrogate and True Waveforms')
plt.savefig(f'{args.results_dir}/Mismatch.pdf', dpi=400)
plt.show()

vmin_f = np.nanmin(mismatch_fine)
vmax_f = np.nanmax(mismatch_fine)
if math.isnan(vmin_f) or math.isnan(vmax_f):
    vmin_f, vmax_f = eps, 1.0
else:
    vmin_f = max(eps, vmin_f)
    vmax_f = max(vmin_f * 10.0, vmax_f)

plt.pcolormesh(
    Q_fine, Chi_fine, mismatch_fine,
    shading='auto',
    cmap='viridis',
    norm=colors.LogNorm(vmin=vmin_f, vmax=vmax_f)
)
plt.colorbar(label='Mismatch (log scale)')
plt.xlabel('Mass Ratio q')
plt.ylabel('Spin χ')
plt.title('Mismatch between Surrogate and True Waveforms (interpolated)')
plt.savefig(f'{args.results_dir}/Interpolated_mismatch.pdf', dpi=400)
plt.show()


