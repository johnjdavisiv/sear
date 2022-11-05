"""

SEAR: Speed estimation algorithm for running

#JJD

""" 

import numpy as np
from scipy import stats, signal, integrate
from functools import partial #For declaring partial functions
import antropy #Raphael V's entropy functions
import time #Just timing


def window_raw_data(v, w_len = 10, fs = 100):
    """
    Take raw accelerometer data and window it, with clever trick for spillvoer
    (i.e. when vm_shape[0] % (w_len*fs) != 0)

    Parameters
    ----------
    vm : numpy 1d array
        Vector magnitude acceleration, in gs. All data expected to be from running.
    w_len : int, optional
        window length in seconds. The default is 10.
    fs : int, optional
        Sample frequency. The default is 100.

    Returns
    -------
    X_features, col_names
    
    X_features is n x p, where p is number of features (depends on options)
    
    col_names is list of length p, with string containing each feature name. 

    """
    
    extra_samples = v.shape[0] % (w_len*fs)
    end_ix = v.shape[0] - extra_samples
    last_window = v[-fs*w_len:].T
    
    #Has an extra window that should only be used to mark the last extra_samples datapoints
    v_windowed = np.concatenate((v[:end_ix].reshape((-1,w_len*fs)), 
                                 last_window), axis=0)
        
    return v_windowed, extra_samples



def extract_features(X_raw, samp_freq = 100, 
                     prepend = 'R',
                     slow_features = True,
                     complexity_features = True,
                     freq_features = True):
    
    """
    Extract time, frequency, and complexity-domain features from raw accelerometer data.
    
    Parameters
    ----------
    
    X_raw : numpy 2d array 
        Resultant acceleration data, in g-units. Should be n x fs*w_len --> 100*10 in SEAR paper
        All data expected to be from running. 
        
    fs : int, optional
        Sampling frequency, in Hz. The default is 100.
    
    prepend : string
        Single-character identifier, "R" for resultant, in SEAR paper
        
    slow_features: bool
        Extract slow-to-calculate complexity features? Default is true. 
        
    complexity_features: bool
        Extract complexity-domain features? Default is true.
        
    freq_features: bool
        Extract frequency-domain features? Default is true. 
        
    
    Returns
    -------
    v_windowed, extra_samples

    """
    
    #Will maybe throw an error if slow features is false and complexity features is false
    start = time.time()
    
    if slow_features:
        print("Extracting all features, this could take several minutes...")
    else:
        print("Extracting only fast features, should take a couple minutes...")
    
    #Quantile wrapper
    def quantile_wrapper(X, *args, **kwargs):
        return np.quantile(X, *args, **kwargs).T
    
    def logp1_moment5(X):
        return np.log(stats.moment(X, moment=5, axis=1) + 1)
    
    #Rate of zero crossings, in crosses per 1/fs (i.e. per sample)
    def array_zero_cross_rate(X):
        #Mean center each row
        Xc = X - np.mean(X, axis=1)[:, np.newaxis]
        #Get zero crossing rate
        zero_cross = np.diff(Xc > 0, axis=1).astype('int').sum(axis=1)/X.shape[1]    
        return zero_cross
    
    # Autocorrelation wrappers
    def autocorr_at_lags(x, lags):
        ac_lags = np.correlate(x, x, mode='full')[x.shape[0] - 1:][lags]
        return ac_lags
    
    def array_acf_lags(X, *args, **kwargs):
        return np.apply_along_axis(autocorr_at_lags, axis=1, arr=X, *args, **kwargs)
    
    #Wrappers for entropy functions
    def array_perm_entropy(X, *args, **kwargs):
        return np.apply_along_axis(antropy.perm_entropy, axis=1, arr=X, *args, **kwargs)
    
    def array_spectral_entropy(X, *args, **kwargs):
        #Requires sample frequency! 
        return np.apply_along_axis(antropy.spectral_entropy, axis=1, arr=X, *args, **kwargs)
    
    def array_svd_entropy(X, *args, **kwargs):
        return np.apply_along_axis(antropy.svd_entropy, axis=1, arr=X, *args, **kwargs)
    
    def array_app_entropy(X, *args, **kwargs):
        return np.apply_along_axis(antropy.app_entropy, axis=1, arr=X, *args, **kwargs)
    
    def array_sample_entropy(X, *args, **kwargs):
        return np.apply_along_axis(antropy.sample_entropy, axis=1, arr=X, *args, **kwargs)
    
    def array_lziv_complexity(X, *args, **kwargs):
        #Must convert to string of 0s and 1s
        X_str = (X > np.mean(X,axis=1, keepdims=True)).astype(int)
        return np.apply_along_axis(antropy.lziv_complexity, axis=1, arr=X_str, *args, **kwargs)
    
    #Wrappers for fractal functions
    def array_petrosian_fd(X, *args, **kwargs):
        return np.apply_along_axis(antropy.petrosian_fd, axis=1, arr=X, *args, **kwargs)
    
    def array_katz_fd(X, *args, **kwargs):
        return np.apply_along_axis(antropy.katz_fd, axis=1, arr=X, *args, **kwargs)
    
    def array_higuchi_fd(X, *args, **kwargs):
        return np.apply_along_axis(antropy.higuchi_fd, axis=1, arr=X, *args, **kwargs)
    
    def array_detrended_fluctuation(X, *args, **kwargs):
        return np.apply_along_axis(antropy.detrended_fluctuation, axis=1, arr=X, *args, **kwargs)

    #---------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    #                              FFT based functions 
    
    def get_dom_freq(X, fs):
        #Do FFt after mean-centering
        X_mags = np.abs(np.fft.rfft(X - np.mean(X, axis=1, keepdims=True), axis=1))
        freqs = np.fft.rfftfreq(X.shape[1], d=1/fs)
        
        #Return freq of dominant frequency in each row
        dom_freqs = freqs[np.argmax(X_mags, axis=1)]
        return dom_freqs
        
    #Actually should be separate functions
    def get_dom_freq_mag(X, fs):
        #Do FFt after mean-centering
        X_mags = np.abs(np.fft.rfft(X - np.mean(X, axis=1, keepdims=True), axis=1))
        #Return freq and mag at the dominant frequency
        dom_freq_mags = np.amax(X_mags, axis=1)
        return dom_freq_mags
        
    def count_fft_peaks(X,fs, pct_max = 0.1):
        #Count the number of peaks of >= pct_max  x magnitude of dom peak in FFT spectrum
        X_mags = np.abs(np.fft.rfft(X - np.mean(X, axis=1, keepdims=True), axis=1))
        n_peaks = np.zeros(X_mags.shape[0])
        for i in range(X_mags.shape[0]):
            pks, pk_info = signal.find_peaks(X_mags[i,:], height=pct_max*np.max(X_mags[i,:]))
            n_peaks[i] = len(pks)
        return n_peaks
    
    # --- BANDPOWER ---
    def bandpower(data, sf, band, window_sec=None, relative=False):
            # Adapted from Raphael Vallat
        """Compute the average power of the signal x in a specific frequency band.
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.
        Return
        ------
        bp : float
            Absolute or relative band power.
        """
        band = np.asarray(band)
        low, high = band
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        # Compute the modified periodogram (Welch)
        freqs, psd = signal.welch(data, sf, nperseg=nperseg)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        
        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule.
        bp = integrate.trapz(psd[idx_band], dx=freq_res)
    
        if relative:
            bp /= integrate.trapz(psd, dx=freq_res)
            
        return bp
    
    #Should be faster for multiple bins
    def multi_bandpower_bins(data, sf, max_bin=15, window_sec=3, relative=False):
        # Adapted from Raphael Vallat
        nperseg = window_sec * sf
    
        # Compute the modified periodogram (Welch)
        freqs, psd = signal.welch(data, sf, nperseg=nperseg)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        
        # Find closest indices of band in frequency vector
        ix_band_i = [np.logical_and(freqs >= i, freqs < i+1) for i in range(max_bin)]
        #Note this was originally freqs<= i+1 but I think it should be inclusive, exclusive
        #To follow conventions and to avoid excessively high correllation
        
        # Integral approximation of the spectrum using Simpson's rule.
        bp_i = [integrate.trapz(psd[this_band], dx=freq_res) for this_band in ix_band_i]
    
        if relative:
            total_pow = integrate.trapz(psd, dx=freq_res)
            bp_rel = [this_bp / total_pow for this_bp in bp_i]
            return bp_rel
        else:
            return bp_i
        
    def log_array_bandpower(X, *args, **kwargs):
        return np.log(np.apply_along_axis(bandpower, 
                                          axis=1, arr=X, *args, **kwargs))
    
    def log_array_multi_bandpower_bins(X, *args, **kwargs):
        return np.log(np.apply_along_axis(multi_bandpower_bins, 
                                          axis=1, arr=X, *args, **kwargs))
    
    # ---------------  Define feature dict --------------------------
    
    #Create dictionary of functions with some arguments "frozen" in place
    function_dict = {'mean':partial(np.mean, axis=1),
               'median':partial(np.median, axis=1),
               'max':partial(np.amax,axis=1),
               'min':partial(np.amin,axis=1),
               'sd':partial(np.std, axis=1),
               'quantiles':partial(quantile_wrapper,q=[0.05,0.25,0.75,0.95], axis=1),
               'kurtosis':partial(stats.kurtosis, axis=1),
               'skew':partial(stats.skew, axis=1),
               'logp1_moment5':partial(logp1_moment5),
               'MAD':partial(stats.median_abs_deviation, axis=1),
               'rms':partial(lambda x: np.sqrt(np.mean(x**2, axis=1))),
               'log_coefvar':partial(lambda x: np.log(np.mean(x,axis=1) / np.std(x,axis=1))),
               'zero_cross_rate':partial(array_zero_cross_rate),
               # --------- AUTOCORRELATION MEASURES ------
               #Autocorr lag at 1, 2, 5, 10, ... samples
               'autocorr_lags':partial(array_acf_lags, lags = [1,2,5,10,25,50,100,150,200]),
               # -------- ENTROPY MEASURES -----------
               'perm_entropy': partial(array_perm_entropy),
               'spectral_entropy':partial(array_spectral_entropy, sf=samp_freq, method='welch'),
               'svd_entropy':partial(array_svd_entropy),
               'app_entropy':partial(array_app_entropy), #SLOW, adds ~120 s
               'sample_entropy':partial(array_sample_entropy), #actually not bad, +20sec
               'lziv_complexity':partial(array_lziv_complexity, normalize=True), #also SLOWW! adds 80s
               'petrosian_fd':partial(array_petrosian_fd),
               'katz_fd':partial(array_katz_fd),
               'higuchi_fd':partial(array_higuchi_fd),
               'detrended_fluctuation':partial(array_detrended_fluctuation),
               # --------- FREQUENCY MEAUSURES --------
               'dominant_freq':partial(get_dom_freq, fs=samp_freq),
               'mag_at_dom_freq':partial(get_dom_freq_mag, fs=samp_freq),
               'n_fft_peaks':partial(count_fft_peaks, fs=samp_freq, pct_max=0.1),
               # ---- BANDPOWER PARAMETERS ------
               'log_bp_06_2_5_abs':partial(log_array_bandpower, sf=samp_freq, band=[0.6, 2.5], 
                                       window_sec=3, relative=False),
               'log_bp_03_15_abs':partial(log_array_bandpower, sf=samp_freq, band=[0.3,15],
                                          window_sec=3, relative=False),
               'log_abs_power_1Hz_bins':partial(log_array_multi_bandpower_bins, 
                                            max_bin=25, sf=samp_freq, 
                                            window_sec=3, relative=False)}
    #If you add app_entropy and lziv complexity you add several minutes (total of 4 minutes)
    #less than 1min if you include everything but those two

    if not slow_features:
        del function_dict['app_entropy']
        del function_dict['lziv_complexity']
   
    if not complexity_features:
        del function_dict['perm_entropy']
        del function_dict['spectral_entropy']
        del function_dict['svd_entropy']
        del function_dict['app_entropy']
        del function_dict['sample_entropy']
        del function_dict['lziv_complexity']
        del function_dict['petrosian_fd']
        del function_dict['katz_fd']
        del function_dict['higuchi_fd']
        del function_dict['detrended_fluctuation']
        
    if not freq_features:
        del function_dict['dominant_freq']
        del function_dict['mag_at_dom_freq']
        del function_dict['n_fft_peaks']
        del function_dict['log_bp_06_2_5_abs']
        del function_dict['log_bp_03_15_abs']
        del function_dict['log_abs_power_1Hz_bins']

    #This is extremely slick python here
    feature_list = [myfun(X_raw) for myname, myfun in function_dict.items()]
    X_features = np.column_stack(feature_list)
    n_feature_cols = [xo.shape[1] if len(xo.shape) > 1 else 1 for xo in feature_list]
    all_features = list(function_dict.keys())

    #get headers for columns
    col_names = []

    for i, feature_name in enumerate(all_features):
        if n_feature_cols[i] == 1:
            col_names.append(prepend + '_' + feature_name)
        else:
            for i in range(n_feature_cols[i]):
                col_names.append(prepend + '_' 
                                 + feature_name + '_' + str(i+1))
    #Should be good
        
    end = time.time()
    print("Finished! Time elapsed: {} sec".format(end - start))
    
    return (X_features, col_names)
    