import os
import h5py
import multiprocessing

import numpy as np
from scipy.interpolate import UnivariateSpline

from surrogw.modules.constants import *
from surrogw.modules.gw_utils import generate_fd_waveform, get_amp_phase, generate_sparse_grid

class GWDataGen:
    def __init__(self, f_lower=f_lower, delta_t=delta_t, window_type=window_type, LAL_taper_method=LAL_taper_method, epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end, f_min_mask=f_min_mask, f_max_mask=f_max_mask, dataset_dir='training_data'):
        self.f_lower = f_lower
        self.delta_t = delta_t
        self.window_type = window_type
        self.LAL_taper_method = LAL_taper_method
        self.epsilon = epsilon
        self.num_extrema_start = num_extrema_start
        self.num_extrema_end = num_extrema_end
        self.f_min_mask = f_min_mask
        self.f_max_mask = f_max_mask

        self.dataset_dir = dataset_dir

        # if not os.path.exists(self.dataset_dir):
        #     os.makedirs(self.dataset_dir)

    @staticmethod
    def process_waveform(params, f_lower, delta_t, window_type, LAL_taper_method, epsilon, num_extrema_start, num_extrema_end, f_min_mask, f_max_mask):
        
        freqs, h_fd = generate_fd_waveform(params=params, f_lower=f_lower, delta_t=delta_t, window_type=window_type, LAL_taper_method=LAL_taper_method, padding_type='power_of_2', epsilon=epsilon, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end)
        mask = (freqs >= f_min_mask) & (freqs <= f_max_mask)
        freqs_masked = freqs[mask]

        amp, phase = get_amp_phase(freqs_masked, h_fd[mask])
        df = freqs_masked[1] - freqs_masked[0]
        norm = np.sqrt(np.sum(amp**2) * df)
        
        return (amp / norm, norm, phase, freqs_masked, params)
    
    def generate_waveforms(self):
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

        self.param_grid_q, self.param_grid_chi = np.meshgrid(q_vals, chi_vals)
        params_list = [{'q': q, 'chi': chi} for q, chi in zip(self.param_grid_q.flatten(), self.param_grid_chi.flatten())]

        args_for_starmap = [(p, self.f_lower, self.delta_t, self.window_type, self.LAL_taper_method, self.epsilon, self.num_extrema_start, self.num_extrema_end, self.f_min_mask, self.f_max_mask) for p in params_list]

        print("Step I: Generating training data...")

        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(GWDataGen.process_waveform, args_for_starmap)

        valid_results = [r for r in results if r is not None]
        self.raw_amps, self.amp_norms, self.raw_phases, self.raw_freqs, self.valid_params = zip(*valid_results)

        print(f"Generated {len(self.raw_amps)} valid waveforms.")
    
    def gen_sparse_freq_grid_and_interpolate(self):
        print("Step II: Creating sparse grids and interpolating...")

        self.f_min_grid = self.f_min_mask
        self.f_max_grid = self.f_max_mask

        self.sparse_freq_amp = generate_sparse_grid(self.f_min_grid, self.f_max_grid, num_points=200, power=1.4) 
        self.sparse_freq_phase = generate_sparse_grid(self.f_min_grid, self.f_max_grid, num_points=200, power=4/3)

        A_mat = np.empty((len(self.sparse_freq_amp), len(self.raw_amps)), dtype=np.float32)
        Phi_mat = np.empty((len(self.sparse_freq_phase), len(self.raw_phases)), dtype=np.float32)

        for i, (amp, phase, freqs) in enumerate(zip(self.raw_amps, self.raw_phases, self.raw_freqs)):

            spline_amp = UnivariateSpline(freqs, amp, s=0, k=3, ext=2)
            spline_phase = UnivariateSpline(freqs, phase, s=0, k=3, ext=2)

            # fmin, fmax = freqs[0], freqs[-1]

            A_mat[:, i] = spline_amp(self.sparse_freq_amp).astype(np.float32)
            Phi_mat[:, i] = spline_phase(self.sparse_freq_phase).astype(np.float32)

        self.A_mat = A_mat
        self.Phi_mat = Phi_mat

    def __call__(self):
        
        self.generate_waveforms()
        self.gen_sparse_freq_grid_and_interpolate()

        with h5py.File(f"{self.dataset_dir}.hdf5", "w") as f:
            f.create_dataset("A_mat", data=np.asarray(self.A_mat), compression="gzip")
            f.create_dataset("Phi_mat", data=np.asarray(self.Phi_mat), compression="gzip")

            f.create_dataset("amp_norms", data=np.asarray(self.amp_norms), compression="gzip")
            f.create_dataset("sparse_freq_amp", data=self.sparse_freq_amp)
            f.create_dataset("sparse_freq_phase", data=self.sparse_freq_phase)

            f.create_dataset("q", data=[p["q"] for p in self.valid_params])
            f.create_dataset("chi", data=[p["chi"] for p in self.valid_params])

            f.create_dataset("param_grid_q", data=np.asarray(self.param_grid_q), compression="gzip")
            f.create_dataset("param_grid_chi", data=np.asarray(self.param_grid_chi), compression="gzip")

            f.attrs["f_lower"] = f_lower
            f.attrs["delta_t"] = delta_t
            f.attrs["f_min_mask"] = f_min_mask
            f.attrs["f_max_mask"] = f_max_mask
