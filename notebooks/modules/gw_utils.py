import numpy as np
from scipy.special import expit

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
