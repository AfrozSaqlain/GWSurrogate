







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