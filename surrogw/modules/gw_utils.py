import numpy as np
from scipy.special import expit
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq

from pycbc.waveform import get_td_waveform

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

    # x = L/(L-n) - L/n
    # w[-(L-1):] = expit(-x)
    # w[-1] = 0.0

    return w

# -----------------------------------------------------------------------------
# ## Waveform Generator
# -----------------------------------------------------------------------------
def generate_fd_waveform(params, f_lower, delta_t, window_type='lal_planck', LAL_taper_method='LAL_SIM_INSPIRAL_TAPER_STARTEND', padding_type='power_of_2', epsilon=0.05, num_extrema_start=32, num_extrema_end=32):
    q = params['q']
    chi = params['chi']
    M_total = 40.0
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
        window = Planck_window_LAL(h_td_raw, taper_method=LAL_taper_method, num_extrema_start=num_extrema_start, num_extrema_end=num_extrema_end) 
    else:
        window = np.ones(len(h_td_raw))

    h_td_windowed = h_td_raw * window
    
    L = len(h_td_windowed)
    
    if padding_type == "power_of_2":
      nfft = 2 ** int(np.ceil(np.log2(2 * L)))
    elif padding_type == 'double':
      nfft = 2 * L

    h_td = np.pad(h_td_windowed, (0, nfft - L))

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