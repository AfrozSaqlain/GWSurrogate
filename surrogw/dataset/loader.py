import h5py

class DataLoader:
    def __init__(self, filename):
        self.load_data(filename)

    def load_data(self, filename):
        with h5py.File(filename, "r") as f:
            self.A_mat = f["A_mat"][:]
            self.Phi_mat = f["Phi_mat"][:]

            self.amp_norms = f["amp_norms"][:]

            self.sparse_freq_amp = f["sparse_freq_amp"][:]
            self.sparse_freq_phase = f["sparse_freq_phase"][:]

            self.q = f["q"][:]
            self.chi = f["chi"][:]

            self.param_grid_q = f["param_grid_q"][:]
            self.param_grid_chi = f["param_grid_chi"][:]

            self.f_lower = f.attrs["f_lower"]
            self.delta_t = f.attrs["delta_t"]
            self.f_min_mask = f.attrs["f_min_mask"]
            self.f_max_mask = f.attrs["f_max_mask"]

        self.n_waveforms = len(self.q)
        self.n_amp_nodes = len(self.sparse_freq_amp)
        self.n_phase_nodes = len(self.sparse_freq_phase)

    def __len__(self):
        return self.n_waveforms