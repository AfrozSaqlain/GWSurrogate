import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

import pycbc
import pycbc.psd

from surrogw.modules.constants import *
from surrogw.dataset.loader import DataLoader
from surrogw.modules.gw_utils import generate_fd_waveform, get_amp_phase, generate_sparse_grid

class QCircFD:
    def __init__(self, data_path, rank_a = 10, rank_p = 10):
        self.rank_a = rank_a
        self.rank_p = rank_p

        self.data = DataLoader(filename=data_path)

    # -----------------------------------------------------------------------------
    # ## Step I: Compute Reduced Bases via SVD
    # -----------------------------------------------------------------------------
    def compute_reduced_basis_svd(self):
        print("Step I: Performing SVD to find reduced bases...")

        Ua, sa, Vta = np.linalg.svd(self.data.A_mat, full_matrices=False)
        Up, sp, Vtp = np.linalg.svd(self.data.Phi_mat, full_matrices=False)

        self.B_a = Ua[:, :self.rank_a]
        self.B_p = Up[:, :self.rank_p]

    # -----------------------------------------------------------------------------
    # ## Step IV: Interpolate Projection Coefficients
    # -----------------------------------------------------------------------------
    def interpolate_projection_coefficient(self):
        print("Step II: Interpolating projection coefficients...")

        Ca = self.B_a.T @ self.data.A_mat
        Cp = self.B_p.T @ self.data.Phi_mat

        q_unique = np.unique(self.data.param_grid_q)
        chi_unique = np.unique(self.data.param_grid_chi)

        self.interpolants_a = []
        for i in range(self.rank_a):
            coeff_grid = Ca[i, :].reshape(len(chi_unique), len(q_unique))
            interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
            self.interpolants_a.append(interp)

        self.interpolants_p = []
        for i in range(self.rank_p):
            coeff_grid = Cp[i, :].reshape(len(chi_unique), len(q_unique))
            interp = RectBivariateSpline(chi_unique, q_unique, coeff_grid, kx=3, ky=3)
            self.interpolants_p.append(interp)

        amp_norms_grid = np.array(self.data.amp_norms).reshape(len(chi_unique), len(q_unique))
        self.interp_amp_norm = RectBivariateSpline(chi_unique, q_unique, amp_norms_grid, kx=3, ky=3)

    # -----------------------------------------------------------------------------
    # ## Step V: Assemble and Evaluate the Surrogate Model
    # -----------------------------------------------------------------------------
    def assemble_and_evaluate_surrogate(self, q_star, chi_star, freqs_out):
        print("Step III: Assembling the surrogate model evaluator.")

        ca_star = np.array([interp(chi_star, q_star)[0, 0] for interp in self.interpolants_a])
        cp_star = np.array([interp(chi_star, q_star)[0, 0] for interp in self.interpolants_p])

        amp_recon_sparse = self.B_a @ ca_star
        phase_recon_sparse = self.B_p @ cp_star

        spline_amp = UnivariateSpline(self.data.sparse_freq_amp, amp_recon_sparse, s=0, k=3, ext=2)
        spline_phase = UnivariateSpline(self.data.sparse_freq_phase, phase_recon_sparse, s=0, k=3, ext=2)

        amp_final = spline_amp(freqs_out)
        phase_final = spline_phase(freqs_out)

        norm_star = self.interp_amp_norm(chi_star, q_star)[0, 0]
        amp_final *= norm_star

        h_fd_recon = amp_final * np.exp(1j * phase_final)
        
        return freqs_out, h_fd_recon
        
    # -----------------------------------------------------------------------------
    # ## Run a test case to validate the surrogate model
    # -----------------------------------------------------------------------------
    def validate_surrogate(self):
        print("\nValidating model with a test waveform...")

        test_params = {'q': 2.3, 'chi': 0.98}

        true_freqs, true_h_fd = generate_fd_waveform(test_params, f_lower, delta_t, window_type=window_type, padding_type='power_of_2', epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end)
        mask = (true_freqs >= self.data.f_min_mask) & (true_freqs <= self.data.f_max_mask)
        true_freqs_masked = true_freqs[mask]
        true_h_fd_masked = true_h_fd[mask]
        true_amp, true_phase = get_amp_phase(true_freqs_masked, true_h_fd_masked)

        surr_freqs, surr_h_fd = self.assemble_and_evaluate_surrogate(test_params['q'], test_params['chi'], true_freqs_masked)
        surr_amp, surr_phase = get_amp_phase(surr_freqs, surr_h_fd)

        # -----------------------------------------------------------------------------
        # ## Compute mismatch
        # -----------------------------------------------------------------------------
        pycbc_surr_h_fd = pycbc.types.FrequencySeries(surr_h_fd, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
        pycbc_true_h_fd = pycbc.types.FrequencySeries(true_h_fd_masked, delta_f=true_freqs_masked[1]-true_freqs_masked[0], epoch=0)
        pycbc_surr_h_fd.start_time = 0
        pycbc_true_h_fd.start_time = 0

        mismatch = 1 - pycbc.filter.matchedfilter.optimized_match(pycbc_surr_h_fd, pycbc_true_h_fd, psd=pycbc.psd.aLIGOZeroDetHighPower(len(pycbc_true_h_fd), pycbc_true_h_fd.delta_f, f_lower), low_frequency_cutoff=self.data.f_min_mask)[0]
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
        plt.savefig(f'Surrogate_Model_vs_True_Model.pdf')
        plt.show()

    # -----------------------------------------------------------------------------
    # ## Save the Surrogate Model
    # -----------------------------------------------------------------------------
    def save_surrogate(self, filename="surrogate_model.hdf5"):
        with h5py.File(filename, "w") as f:
            # Reduced bases
            f.create_dataset("B_a", data=self.B_a, compression="gzip")
            f.create_dataset("B_p", data=self.B_p, compression="gzip")

            # Sparse frequency grids
            f.create_dataset("sparse_freq_amp", data=self.data.sparse_freq_amp )
            f.create_dataset("sparse_freq_phase", data=self.data.sparse_freq_phase )

            # Training parameter grids
            f.create_dataset("param_grid_q", data=self.data.param_grid_q )
            f.create_dataset("param_grid_chi", data=self.data.param_grid_chi )

            # Amplitude normalization
            f.create_dataset("amp_norms", data=self.data.amp_norms )

            # Metadata
            f.attrs["rank_a"] = self.rank_a
            f.attrs["rank_p"] = self.rank_p

            f.attrs["f_lower"] = self.data.f_lower
            f.attrs["delta_t"] = self.data.delta_t
            f.attrs["f_min_mask"] = self.data.f_min_mask
            f.attrs["f_max_mask"] = self.data.f_max_mask

        print(f"Surrogate model saved to {filename}")







