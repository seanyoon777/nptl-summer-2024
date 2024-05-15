# -------------------------------------------------------------------------------------------------
# NeuralFeatureExtractor.py 
# -------------------------------------------------------------------------------------------------
# Functionalities for neural feature extraction, pre-processing and read/write utilities. Optimized 
# for speed and modularity.
# Written by Sean Yoon [sean777@stanford.edu]. 
# Parts of code adapted from Maitreyee Wairagkar and Benyamin Meschede-Krasa. 
# -------------------------------------------------------------------------------------------------
# Created     : 2024-05-03
# Last update : 2024-05-03
# -------------------------------------------------------------------------------------------------
from Globals import *
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# Functions to read and write from ns5 files
# -------------------------------------------------------------------------------------------------

def read_ns5_file(ns5_filename: str, n_channels: int, include_audio: bool=False, audio_channel: int=-1):
    """
    This function reads raw neural data from a ns5 format file, including audio recordings.
    Args:
        ns5_filename  : filename of the ns5 file
        n_channels    : number of channels to be included in analysis 
        include_audio : (optional) boolean indicating whether to include audio recording from file. 
                        defaults to False
        audio_channel : optional, index of channel containing audio data. defaults to -1

    Returns:
        raw_neural : [samples x channels] shape array of raw voltage recordings in int16
        audio      : (optional) vector of audio file with same length as raw_neural. also in int16
    """
    
    # open, read, and close ns5 file
    nsx_file = NsxFile(ns5_filename)
    all_dat = nsx_file.getdata('all', 0) # get all electrodes and start from 0s 
    nsx_file.close()
    
    # data is in last cell of 'data'. we only extract the first n_channel channels
    raw_neural = all_dat['data'][-1][:n_channels,:] 
    
    # extract audio signals from data
    # convert default memmap format to numpy array for faster in-memory access in future processing
    if include_audio:
        audio = all_dat['data'][-1][audio_channel,:]
        return np.asarray(raw_neural.T), np.asarray(audio)
    else: 
        return np.asarray(raw_neural.T)


def unscramble_chans(dat: np.ndarray, ch_to_elec: list=CH_TO_ELECTRODES): 
    """
    Unscrambles neural recording channels. The first and last 64 channels are for the arrays
    implanted in the 6v inferior and superior area, respectively. Used for offline analysis.
    Args:
        dat        : [samples x channels] shape array of raw voltage recordings. 
        ch_to_elec : electrode number corresponding to each channel. default mapping in Globals.py

    Returns:
        unscrambled_dat : [samples x channels] shape array of unscrambled raw voltage recordings in int16. 
    """
    
    unscrambled_dat = np.zeros(shape=dat.shape, dtype='int16')
    for ch in range(len(ch_to_elec)): 
        unscrambled_dat[:,ch_to_elec[ch]] = dat[:,ch]
    return unscrambled_dat
    
    

# -------------------------------------------------------------------------------------------------
# NeuralProcessor class, used for denoising and feature extraction
# -------------------------------------------------------------------------------------------------

class NeuralProcessor: 
    def __init__(self, params): 
        """
        Initializes the NeuralProcessor class to denoise and extract features from neural data. 
        Currently supports various data processing techniques including local field potential, 
        local motor potential, threshold crossing bin count, and spiking bandpower.
        
        Args:
            params (dict) : Configuration parameters for feature extraction, with the following keys: 
                - processes (list)    : Preprocessing steps for denoising. Supported values include 
                  objects ButterworthFilter, ReReferenceFilter, DownSample.
                - thresh_mults (list) : Threshold multipliers for deteftion, such as [-4.5, -4.0, 
                  -3.5]. Defaults to an empty list.
                - thresh_method (str) : Method for calculating thresholds, either 'rms' or 'std'. 
                  Defaults to 'rms'.
                - spike_pow_bands (list): Frequency bands in Hz for spiking bandpower calculations, 
                  e.g., [(100, 500), (400, 1000), (1000, 2500)]. Defaults to an empty list.
                  
                - bin_size (int)         : Bin size in ms for offline sliding window analysis.
                - shift_size (int)       : Bin shift size in ms for offline sliding window analysis.
        
        Raises:
            KeyError: If required parameters ('bin_size' or 'shift_size') are missing in `params`.

        Example: 
            Initializes with custom parameters: 
            NP = NeuralProcessor({
                "processes": [
                    ButterworthFilter("bandpass", [50, 200], ord=4, fs=30000), 
                    ReReferenceFilter("lrr", max_seconds=30)
                    ], 
                "thresh_mults": [-4.5, -4.0, -3.5], 
                "thresh_method": "rms", 
                "spike_pow_bands": [(100, 500), (400, 1000), (1000, 2500)], 
                "bin_size": 20, 
                "shift_size": 20, 
                "fs": 30000
            })
        """
        
        self.processes        = params.get("processes", [])
        self.thresh_mults     = params.get("thresh_mults", [])
        self.thresh_method    = params.get("thresh_method", "rms")
        self.raw_thresholds    = params.get("thresh_values", None)
        self.spike_pow_bands  = params.get("spike_pow_bands", [])
        self.lmp_boxsizes     = params.get("lmp_boxsizes", [])
        self.bin_size         = params.get("bin_size", 0)
        self.shift_size       = params.get("shift_size", 0)
        
        self.TCCExtractor = ThresholdCrossingExtractor(self.thresh_mults, self.bin_size, self.shift_size, self.thresh_method, self.raw_thresholds)
        self.SBPExtractor = SpikePowExtractor(self.spike_pow_bands, self.bin_size, self.shift_size)
        self._LMPExtractor = LMPExtractor(self.lmp_boxsizes) 
    
    
    def __call__(self, dat: np.ndarray, n_arrays: int, n_electrodes: int, fs: int, verbose: bool=False) -> dict: 
        """
        Denoises and extracts features for input data. 
        
        Args:
            dat          : [samples x channels] shape array of neural data
            n_arrays     : number of arrays
            n_electrodes : number of electrodes per array
            fs           : sampling frequency of neural data
            verbose      : prints progress after every denoising / feature extraction step
        Returns:
            out          : dictionary containing extracted features, each following [samples x channels]
        """
        
        lfp, fs_new = self.denoise(dat=dat, n_arrays=n_arrays, n_electrodes=n_electrodes, fs=fs, verbose=verbose) 
        out = self.extract_features(lfp, fs=fs_new, raw_threshold=self.raw_thresholds, verbose=verbose)
        return out
    
    
    def denoise(self, dat: np.ndarray, n_arrays: int, n_electrodes: int, fs: int, verbose: bool=False) -> np.ndarray: 
        """
        Denoises input data. 
        
        Args:
            dat          : [samples x channels] shape array of neural data
            n_arrays     : number of arrays
            n_electrodes : number of electrodes per array
            fs           : sampling frequency of neural data
            verbose      : prints progress after every denoising / feature extraction step
        Returns:
            denoised     : [samples x channels] shape array of denoised data 
            fs_new       : new sampling frequency if modified during signal processing
        """
        
        denoised = dat.copy()
        # create a copy of sampling frequency in case operations affecting sampling frequency (e.g.
        # downsampling) are applied 
        fs_new = fs
        
        if verbose: 
            print("Denoising steps")
        
        for i, process in enumerate(self.processes): 
            denoised = process(dat=denoised, n_arrays=n_arrays, n_electrodes=n_electrodes, fs=fs_new)

            # update frequency stored in NeuralProcessor class if downsampling is applied
            if isinstance(process, Downsampler): 
                fs_new = int(fs_new / process.ds_factor)
            
            if verbose: 
                print(f'[{i+1}/{len(self.processes)}]: {self._print_process(process)}')
                
        return denoised, fs_new
        
        
    def extract_features(self, dat: np.ndarray, fs: int, raw_threshold=None, verbose: bool=False) -> dict: 
        """
        Extracts features from input data. 
        
        Args:
            dat          : [samples x channels] shape array of neural data
            fs           : sampling frequency of neural data
        Returns:
            out          : dictionary containing extracted features, each following [samples x channels]
        """
        
        # internal logic for numbering each feature extraction step – it's good to have this for debugging purposes
        features_num = np.sum([True, 
                               len(self.TCCExtractor.multipliers) > 0, 
                               len(self.SBPExtractor.freq_bands) > 0, 
                               len(self._LMPExtractor.boxsizes) > 0])
        i = 1
        out = {"lfp": dat}
        
        if verbose: 
            print("Feature extraction steps")
            print(f'[{i}/{features_num}] Local field potentials extracted')
            i += 1
        
        if len(self.TCCExtractor.multipliers) > 0: 
            out.update(self.TCCExtractor(dat, fs=fs))
            if verbose: 
                print(f'[{i}/{features_num}] Threshold crossing counts extracted for multipliers {self.TCCExtractor.multipliers}')
                i += 1
        
        if len(self.SBPExtractor.freq_bands) > 0: 
            out.update(self.SBPExtractor(dat, fs=fs))
            if verbose: 
                print(f'[{i}/{features_num}] Spiking bancpower extracted for frequency bands {self.SBPExtractor.freq_bands}Hz')
                i += 1
            
        if len(self._LMPExtractor.boxsizes) > 0: 
            out.update(self._LMPExtractor(dat, fs=fs))
            if verbose: 
                print(f'[{i}/{features_num}] Local motor potentials extracted for box sizes {self._LMPExtractor.boxsizes}ms')
                i += 1
            
        return out
    
    
    def append_processes(self, process: Any) -> None: 
        """Appends denoising step"""
        self.processes.append(process)
    
    
    def remove_process(self, process: Any) -> None:
        """Removes denoising step"""
        if process in self.processes: 
            self.processes.remove(process)

    
    def _print_process(self, process): 
        if isinstance(process, ReReferenceFilter): 
            if process.method.lower() == "lrr": 
                return f"linear regression reference (LRR) filtered"
            elif process.method.lower() == "car": 
                return f"common average reference (CAR) filtered"
        elif isinstance(process, Downsampler): 
            return f"downsampled by factor of {process.ds_factor}"
        elif isinstance(process, ChevyshevFilter): 
            return f"{process.filt_type} filtered (Chevyshev type I) for Wn={process.Wn}Hz"
        elif isinstance(process, ButterworthFilter): 
            return f"{process.filt_type} filtered (Butterworth) for Wn={process.Wn}Hz"
        

    def summary(table_width=70): 
        """
        Produces a summary of the NeuralProcessor
        Args: 
        
        """
        # logic for setting table width and checking whether parameters are valid


    def summary(self, table_width=70): 
        """
        Produces a summary of the LMPExtractor class object
        Args:
            table_width : width of summary table. Defaults to 60 and minimum 50.
        """
        # logic for setting table width and checking whether parameters are valid
        if table_width < 50: 
            raise ValueError("Minimum width is 50")
        max_width = table_width - 2
        
        # prints table header
        print("┌" + "─" * max_width + "┐")
        title = "Neural processor configuration"
        left = int(np.ceil((max_width - len(title)) / 2))
        right = int(np.floor((max_width - len(title)) / 2))
        print("│" + " " * left + title + " " * right + "│")
        print("├" + "=" * max_width + "┤")
        
        # prints denoising table subheader
        processes_title = "Denoising steps:"
        print("| " + processes_title + " " * (max_width - len(processes_title) - 1) + "|")
        print("├" + "─" * max_width + "┤")
        for process in self.processes: 
            pass 
            #process.summary(table_width)
        print("├" + "=" * max_width + "┤")
        
        # prints feature extraction table subheader 
        feature_title = "Output features:"
        print("| " + feature_title + " " * (max_width - len(feature_title) - 1) + "|")
        print("├" + "─" * max_width + "┤")
        if len(self.thresh_mults) > 0: 
            self.TCCExtractor.summary(table_width)
        if len(self.lmp_boxsizes) > 0: 
            self._LMPExtractor.summary(table_width)
            
        print("└" + "─" * max_width + "┘")
        
        # # handle potentially long text for boxcar filter sizes
        # coltext = "│ Boxcar filter size (ms) : " + " |"
        # space_width = table_width - len(coltext)
        # mult_text = chunk_text(str(self.boxsizes), max_width=space_width)
        
        # # print frequency bands line by line
        # for i, line in enumerate(mult_text): 
        #     if i == 0 : 
        #         print_line(f"│ Boxcar filter size (ms) : {line}", max_width)
        #     else: 
        #         print_line(f"│                           {line}", max_width)

        # print_line(f"│ Signal sampling freq    : {self.fs} Hz ", max_width)
        # 
        
        
        
# -------------------------------------------------------------------------------------------------
# Optimized classes and functions for neural signal denoising, and preprocessing prior to feature 
# extraction. 
# Currently supported features & definitions: 
#     > Downsampling              : Downsampling
#     > Frequency-based filtering : Bandpass, lowpass, and highpass filtering
#     > Re-referencing            : Common average referencing (CAR) and linear regression 
#                                   referencing (LRR)
# -------------------------------------------------------------------------------------------------

class Downsampler: 
    def __init__(self, ds_factor: float): 
        """
        Initalizes Downsampler class for downsampling signal. Assumes input data to be already 
        bandpass filtered. 

        Args:
            ds_factor : downsampling factor
        """
        self.ds_factor = ds_factor
    
    
    def __call__(self, dat: np.ndarray, **kwargs): 
        """
        Downsamples input neural data. Assumes input data to be already filtered.
        
        Args: 
            dat : [samples x channels] shape array of neural data
        """
        return dat[::self.ds_factor]


class ChevyshevFilter: 
    def __init__(self, filt_type, Wn, rp, ord, fs, non_causal=True, t_delay=0): 
        # save in case frequency of the signal changes during processing before filter applied
        self.filt_type = filt_type
        self.rp = rp
        self.Wn = Wn
        self.ord = ord
        self.fs = fs
        self.non_causal = non_causal
        self.t_delay = t_delay
        
        # check valid filter type
        if filt_type not in ['bandpass', 'lowpass', 'highpass']:
            raise ValueError("Unsupported filtering method")
        
        # build filter and obtain parameters
        self.params = signal.cheby1(N=ord, Wn=Wn, rp=rp, btype=filt_type, output='sos', fs=fs)
        
        
    def __call__(self, dat, fs, **kwargs): 
        """
        Filters signal using cascaded second-order sections.

        Args:
            dat : [samples x channels] shape array of neural data
            fs  : sampling frequency of neural data
        Returns:
            filtered : [samples x channels] shape array of filtered data
        """
        # if the class is called by calling a NeuralProcessor object, the processes of the NeuralProcessor
        # could've had changed the sampling frequency. We update the sampling frequency correspondingly. 
        if fs != self.fs: 
            self.params = signal.cheby1(N=self.ord, Wn=self.Wn, rp=self.rp, btype=self.filt_type, output='sos', fs=fs)
        
        # return filtered signal depending on causal or non-causal filtering
        if self.non_causal: 
            filtered = signal.sosfiltfilt(self.params, dat, axis=0)
            if self.t_delay != 0: 
                n_samples_delay = int(self.t_delay * fs / 1000)
                filtered = np.roll(filtered, -n_samples_delay)
            return filtered
        return signal.sosfilt(self.params, dat, axis=0)
    
    
class ButterworthFilter:
    def __init__(self, filt_type, Wn, ord, fs, non_causal=True, t_delay=0): 
        """
        Initializes ButterworthFilter class for frequency-based filtering neural data. Currently 
        supports highpass, bandpass, and lowpass filtering. Uses a digital butterworth IIR filter 
        and its second-order sections representation for computation. 
        
        Args:
            filt_type  : The type of filter. One of {'bandpass', 'lowpass', 'highpass'}
            Wn         : the critical frequency or frequencies. For lowpass and highpass filters, 
                         Wn is a scalar; for bandpass and bandstop filters, Wn is a length-2 sequence.
            ord        : order of the butterworth filter. 
            fs         : sampling frequency of neural data. 
            non_causal : non causal (zero-phase) filtering. Defaults to true. 
            t_delay    : time delay to simulate a non-causal system

        Raises:
            ValueError: _description_
        """
        # save in case frequency of the signal changes during processing before bandpass applied
        self.filt_type = filt_type
        self.Wn = Wn
        self.ord = ord
        self.fs = fs
        self.non_causal = non_causal
        self.t_delay = t_delay
        
        # check valid filter type
        if filt_type not in ['bandpass', 'lowpass', 'highpass']:
            raise ValueError("Unsupported filtering method")
        
        # build filter and obtain parameters
        self.params = signal.butter(N=ord, Wn=Wn, btype=filt_type, output='sos', fs=fs)

    
    # @staticmethod
    # @nb.jit(nopython=True)
    # def _delay(dat, n_samples_delay): 
 
 ##TODO: def _delay(): 
        
        
    def __call__(self, dat, fs=0, **kwargs): 
        """
        Filters signal using cascaded second-order sections.

        Args:
            dat : [samples x channels] shape array of neural data
            fs  : sampling frequency of neural data
        Returns:
            filtered : [samples x channels] shape array of filtered data
        """
        # if the class is called by calling a NeuralProcessor object, the processes of the NeuralProcessor
        # could've had changed the sampling frequency. We update the sampling frequency correspondingly. 
        if fs != self.fs and fs > 0: 
            self.params = signal.butter(N=self.ord, Wn=self.Wn, btype=self.filt_type, output='sos', fs=fs)
        
        # return filtered signal depending on causal or non-causal filtering
        if self.non_causal: 
            filtered = signal.sosfiltfilt(self.params, dat, axis=0)
            if self.t_delay != 0: 
                n_samples_delay = int(self.t_delay * fs / 1000)
                filtered = np.roll(filtered, -n_samples_delay)
            return filtered
        return signal.sosfilt(self.params, dat, axis=0)
    
    
    
class ReReferenceFilter: 
    def __init__(self, method: str, max_seconds: int=0, ref_dat: np.ndarray=None, ref_mat: np.ndarray=None): 
        """
        Initalizes the class ReReferenceFilter, a filter for performing re-referencing to eliminate 
        any noise or other common artifacts across neural channels. This class currently supports 
        common average referencing (CAR) or linear regression referencing (LRR) for each array. 
        
        LRR is recommended since unlike CAR, it does not assume equal noise across all channels in 
        an array. For more information, see Young et al (2018): 
        https://iopscience.iop.org/article/10.1088/1741-2552/aa9ee8/pdf
        
        Args:
            method      : re-referencing method, either 'car' or 'lrr' 
            max_seconds : Only for LRR. Length of data for calculating LRR coefficients. 
                          Uses entire data block by default if not specified. 
            ref_dat     : Optional, specific neural data of shape [samples x channels] for computing LRR weights. 
                          Uses input data to ReReferenceFilter if not specified. 
            ref_mat     : Optional, pre-computed LRR weights of shape [channels x channels]. 
        """
        # check that method is valid 
        method = method.lower()
        if method not in ['car', 'lrr']:
            raise ValueError("Unsupported re-referencing method")
        self.method = method 
        
        # set other paramters
        self.max_seconds = max_seconds
        self.ref_dat     = ref_dat
        self.ref_mat     = ref_mat
        
        # handle edge cases where ref_mat and ref_dat are both specified
        if self.ref_mat is not None and self.ref_dat is not None: 
            print('Warning: Data for calculating coefficients specified in addition to pre-calculated coefficients. ')
            print('         Neural data will be ignored. To avoid this, reinitialize with only ref_dat or ref_mat.')
            
    
    def _car(self, dat: np.ndarray, n_arrays: int, n_electrodes: int, **kwargs) -> np.ndarray:
        """
        Applies common average referencing (CAR) to neural data. 
        Args:
            dat          : [samples x channels] shape array of neural data
            n_arrays     : number of arrays
            n_electrodes : number of electrodes per array

        Returns:
            np.ndarray: _description_
        """
        dat      = dat.astype('float32')
        denoised = np.zeros(shape=dat.shape, dtype='float32')
        
        if self.ref_mat is None: 
            if self.ref_dat is None: 
                ref_mat = self.get_weights(dat, n_arrays, n_electrodes)
            else: 
                ref_mat = self.get_weights(self.ref_dat, n_arrays, n_electrodes)
        else: 
            ref_mat = self.ref_mat 
            
        # apply car to each array
        for array in range(n_arrays):
            start = n_electrodes * array 
            end   = n_electrodes * (array + 1)
            dat_array = dat[:,start:end]
            denoised[:,start:end] = self._car_denoise(dat_array, ref_mat[array])
        
        return denoised
    
    
    @staticmethod
    @nb.jit(nopython=True)
    def _get_car_weights(dat_array): 
        '''
        Gets CAR weights from single array neural data 
        '''
        dat_array    = np.ascontiguousarray(dat_array)
        n_samples    = dat_array.shape[0]
        common_noise = np.zeros(shape=(n_samples, 1), dtype='float32')
        # for each timestamp, calculate reference (average of all channels)
        for t in range(n_samples): 
            common_noise[t] = np.mean(dat_array[t,:])
        return common_noise
        
        
    @staticmethod
    @nb.jit(nopython=True)
    def _car_denoise(dat_array, common_noise): 
        '''
        CAR for single array neural data. 
        '''
        dat_array    = np.ascontiguousarray(dat_array)
        common_noise = np.ascontiguousarray(common_noise)
        return dat_array - common_noise
    
    
    def set_weights_as(self, ref_mat): 
        self.ref_mat = ref_mat
        
    
    def set_weights_with(self, dat: np.ndarray, n_array: int, n_electrodes: int, fs: int): 
        self.ref_dat = dat
        self.ref_mat = self.get_weights(dat, n_array, n_electrodes, fs)

        
    def reset_weights(self): 
        self.ref_mat = None
    
    
    def get_weights(self, dat: np.ndarray, n_arrays: int, n_electrodes: int, fs: int=None): 
        '''
        Returns: 
            weights : [arrays x samples x 1] for CAR, [arrays x channels x channels] for LRR
        '''
        n_samples, n_channels = dat.shape
        
        if self.method == "car": 
            weights = np.zeros(shape=(n_arrays, n_samples, 1), dtype='float32')
        elif self.method == "lrr": 
            weights = np.zeros(shape=(n_arrays, n_channels, n_channels), dtype='float32')
            
        for array in range(n_arrays): 
            start = n_electrodes * array 
            end   = n_electrodes * (array + 1)
            dat_array = dat[:,start:end]
            if self.method == "car": 
                weights[array] = self._get_car_weights(dat_array)
            elif self.method == "lrr": 
                weights[array] = self._get_lrr_weights(dat_array, fs=fs, max_seconds=self.max_seconds)
        
        return weights
            
        
    @staticmethod 
    @nb.jit(nopython=True) 
    def _get_lrr_weights(dat: np.ndarray, fs: int, max_seconds: int=0) -> np.ndarray: 
        """
        Calculates LRR weights for neural data from single array. 
        Args:
            dat         : [samples x channels] shaped neural data from single array
            max_seconds : maximum number of seconds (s) for LRR weight calculation
            fs          : sampling frequency of neural data

        Returns:
            ref_mat : [channels x channels] LRR weights for single array
        """
        n_samples, n_channels = dat.shape
        dat     = np.ascontiguousarray(dat.astype('float32'))
        ref_mat = np.zeros(shape=(n_channels, n_channels), dtype='float32')
        
        '''
        Subsample data to use for LRR weight calculation.
        Randomize the order of the data on the time axis to avoid biasing the LRR weights.
        Only use up to max_seconds of data. If the data is less than max_seconds or if max_seconds
        is not specified, use the entire data block. 
        Note that subsampling and randomization is only done for weight calculation. LRR is later 
        applied to all data.
        '''
        # if max_seconds is not specified, use the entire data block
        if max_seconds == 0: 
            dat_sample = dat
            max_idx = n_samples
        # if specified, subsample data
        else: 
            sample_len = max_seconds * fs
            max_idx    = min(sample_len, n_samples)
            rand_idx   = np.random.permutation(np.arange(n_samples))
            use_idx    = rand_idx[0:max_idx]
            dat_sample = dat[use_idx,:]
            
        for ch in range(n_channels): 
            '''
            Here is where the LRR weights actually get calculated.
            For each channel, we are calculating the weights of all other channels to be later 
            subtracted. 
            We do this by solving the equation:
            Y = X*W
            where Y is the data from the channel we are calculating weights for, X is the data from
            all other channels, and W is the weight matrix.
            Using least squares, we get the expression  
            W = inv(X.T X) X.T Y
            Repeat this for every channel. Resultant weight matrix is ref_mat, of size 
            [n_channels x n_channels].
            '''
            
            # get a list of all channel indices excluding the current one 
            X = np.zeros(shape=(max_idx, n_channels - 1), dtype='float32')
            X[:,ch:] = dat_sample[:,ch+1:]
            X[:,:ch] = dat_sample[:,:ch]
            X = np.ascontiguousarray(X)
            y = np.ascontiguousarray(dat_sample[:,ch])
            
            # solve the optimized least squares to get weights for this channel
            weights = lstsq_pseudoinverse(X, y)
            
            # Add the weights to the larger weight matrix of all channels in appropriate positions, 
            # leaving space for the current channel where the weight is zero
            ref_mat[ch,:ch]   = weights[:ch] 
            ref_mat[ch,ch+1:] = weights[ch:] 
            
        return ref_mat
    
    
    @staticmethod 
    @nb.jit(nopython=True) 
    def _lrr_denoise(dat: np.ndarray, ref_mat: np.ndarray) -> np.ndarray: 
        """
        Denoises neural data from single array given LRR weights. 

        Args:
            dat     : [samples x channels] shaped neural data from single array
            ref_mat : [channels x channels] shaped LRR weights for single array

        Returns:
            denoised : [samples x channels] shaped denoised data from single array
        """
        dat_array = np.ascontiguousarray(dat)
        ref_mat = np.ascontiguousarray(ref_mat)
        return dat_array - np.dot(dat_array, ref_mat)
    
    
    def _lrr(self, dat: np.ndarray, n_arrays: int, n_electrodes: int, fs: int) -> np.ndarray: 
        """
        Denoises neural data using LRR. 
        Args:
            dat          : [samples x channels] shape array of neural data
            n_arrays     : total number of arrays
            n_electrodes : number of electrodes per array
            fs           : sampling frequency of neural data
        Returns:
            denoised : [samples x channels] shape of re-referenced data in float32 
        """
        dat = dat.astype('float32')
        denoised = np.zeros(shape=dat.shape, dtype='float32') 

        # retrieve reference matrix. If reference matrix do not exist but reference data is given, 
        # compute and save reference matrix. 
        if self.ref_mat is None: 
            if self.ref_dat is None: 
                ref_mat = self.get_weights(dat, n_arrays, n_electrodes, fs)
            else: 
                ref_mat = self.get_weights(self.ref_dat, n_arrays, n_electrodes, fs)
        else: 
            ref_mat = self.ref_mat
            
        # iterate over each array to denoise 
        for array in range(n_arrays): 
            start = n_electrodes * array 
            end   = n_electrodes * (array + 1)
            dat_array = dat[:, start:end]
            denoised[:, start:end] = self._lrr_denoise(dat_array, ref_mat=ref_mat[array])
            
        return denoised
    
       
    def __call__(self, dat: np.ndarray, n_arrays: int, n_electrodes: int, fs: int, **kwargs) -> np.ndarray: 
        """
        Returns re-referenced signal using car or lrr. 
        Args:
            dat          : [sample x channel] shape array of neural data
            n_arrays     : total number of arrays
            n_electrodes : number of electrodes per array
        Returns:
            denoised     : [sample x channel] shape array of re-referenced data in float32 
        """
        # return different result based on method (car or lrr)
        if self.method == 'car': 
            denoised = self._car(dat, n_arrays, n_electrodes, fs)
        elif self.method == 'lrr': 
            denoised = self._lrr(dat, n_arrays, n_electrodes, fs)
        return denoised


# -------------------------------------------------------------------------------------------------
# Optimized classes and functions for neural feature extraction. 
# Currently supported fetures & definitions: 
#     > Local field potential (LFP)   : Denoised data via the pipeline above is equivalent to LFP.
#     > Local motor potential (LMP)   : Lower frequency components of local field potentials. Refer 
#                                       to Stavisky et al (2015) for more detail.
#                                       https://iopscience.iop.org/article/10.1088/1741-2560/12/3/036009/pdf
#     > Threshold crossing count/rate : Sum or average of threshold crossings for each time bin. 
#     > Spiking Bandpower (SBP)       : Power of signal for each time bin. 
# -------------------------------------------------------------------------------------------------


class LMPExtractor: 
    def __init__(self, boxsizes): 
        """
        Initializes the LMPExtractor class, used to extract local motor potentials from local field 
        potential data. Local motor potentials are obtained by boxcar filtering the LFP, which is 
        equivalent to a digital lowpass filter. For more detail, refer to Shenoy et al (2015): 
        https://iopscience.iop.org/article/10.1088/1741-2560/12/3/036009/pdf 

        Args:
            boxsizes : list of boxsizes in ms
            fs       : sampling frequency of neural data
        """
        self.boxsizes = boxsizes 
        # if boxsizes is given as a single integer – this is for user convenience
        if not isinstance(boxsizes, list): 
            self.boxsizes = [boxsizes]
        self.fs = 0
    
    
    @staticmethod
    @nb.njit(parallel=True)
    def _compute_boxcar(dat, boxsize, fs): 
        """
        Optimized function for boxcar filtering. Parallelization is used for faster processing, 
        since np.convolve is quite slow. 

        Args:
            dat     : [samples x channels] shape array of neural data
            boxsize : boxcar filter size in ms
            fs      : sampling frequency of neural data
        Returns:
            boxcar_dat : [samples x channels] shape array of boxcar filtered data 
        """
        # Initialize variables and boxcar filter
        dat = dat.astype('float32')
        boxsize_cnt = int(boxsize * fs / 1000) 
        window = np.ascontiguousarray(np.ones(boxsize_cnt, dtype='float32') / boxsize_cnt)
        boxcar_dat = np.empty(shape=dat.shape, dtype='float32')
        
        # Iterate over channels and convolve with boxcar filter
        for ch in nb.prange(dat.shape[1]): 
            ch_dat = np.ascontiguousarray(dat[:, ch])
            boxcar_dat[:, ch] = np.convolve(ch_dat, window, mode='same')
    
        return boxcar_dat
    
    
    def summary(self, table_width=60): 
        """
        Produces a summary of the LMPExtractor class object
        Args:
            table_width : width of summary table. Defaults to 60 and minimum 50.
        """
        # logic for setting table width and checking whether parameters are valid
        if table_width < 50: 
            raise ValueError("Minimum width is 50")
        max_width = table_width - 2
        
        # prints table header
        print("┌" + "─" * max_width + "┐")
        title = "Local motor potential extractor configuration"
        left = int(np.ceil((max_width - len(title)) / 2))
        right = int(np.floor((max_width - len(title)) / 2))
        print("│" + " " * left + title + " " * right + "│")
        print("├" + "─" * max_width + "┤")
        
        # handle potentially long text for boxcar filter sizes
        coltext = "│ Boxcar filter size (ms) : " + " |"
        space_width = table_width - len(coltext)
        mult_text = chunk_text(str(self.boxsizes), max_width=space_width)
        
        # print frequency bands line by line
        for i, line in enumerate(mult_text): 
            if i == 0 : 
                print_line(f"│ Boxcar filter size (ms) : {line}", max_width)
            else: 
                print_line(f"│                           {line}", max_width)

        print_line(f"│ Signal sampling freq    : {self.fs} Hz ", max_width)
        print("└" + "─" * max_width + "┘")
    
        
    def __call__(self, dat, fs, **kwargs): 
        """
        Extracts local motor potentials from the input data based on specified configuration
        parameters. 
        
        Args:
            dat : [samples x channels] shape array of neural data
        Returns:
            out : dict of boxcar filtered signal for each box size, each following 
                  [samples x channels] shape
        """
        out = {} 
        self.fs = fs
        # iterate over each boxsize and boxcar filter signal
        for boxsize in self.boxsizes: 
            out[f"lmp_{boxsize}"] = self._compute_boxcar(dat, boxsize, self.fs)
            
        return out 


class ThresholdCrossingExtractor: 
    # TODO: implement threshold crossing rate too
    def __init__(self, multipliers, bin_size, shift_size, method="rms", raw_thresholds=None): 
        """
        Initializes the ThresholdCrossingExtractor class used for counting threshold crossing 
        instances in a specified bin. Used for offline sliding window analysis. 
        
        Args:
            multipliers : list of threshold multipliers. also accepts int or float (usually -4.5)
            bin_size    : size of the bin in ms
            shift_size  : size of the shift between bins in ms
        """
        self.multipliers = multipliers
        # contain input multiplier in list if it is single number – this is for user convenience
        if not isinstance(multipliers, list): 
            self.multipliers = [multipliers]
            
        self.bin_size = bin_size 
        self.shift_size = shift_size
        self.fs = 0
        self.raw_thresholds = raw_thresholds
        
        # check whether method is valid
        if method not in ["rms", "std"]: 
            raise ValueError("Unsupported thresholding method")
        self.method = method
        
        
    @staticmethod
    @nb.jit(nopython=True)
    def get_raw_threshold(dat, method="rms"): 
        """
        Calculates the threshold for each channel before multipliying with specified factor. 

        Args:
            dat    : [samples x channels] shape array of neural data
            method : method used for calculating threshold values, one of {"rms" or "std}. Defaults to "rms"
        Returns:
            raw_threshold : [channels] shape vector of threshold before multiplying
        """
        dat = dat.astype('float32')
        
        # check whether method is valid
        if method not in ["rms", "std"]:
            raise ValueError("Unsupported thresholding method")
        
        n_channels = dat.shape[1]
        raw_threshold = np.zeros(n_channels, dtype='float32')
        
        # iterate over each channel to obtain threshold value
        for ch in range(n_channels): 
            if method == "rms": 
                raw_threshold[ch] = np.sqrt(np.mean(np.square(dat[:,ch])))
            elif method == "std": 
                raw_threshold[ch] = np.std(dat[:,ch])
                
        return raw_threshold


    @staticmethod
    @nb.jit(nopython=True)
    def count_threshold_crossings(dat, raw_threshold, mult, bin_size, shift_size, fs): 
        """
        Counts threshold crossings for given multiplier, bin size, and shift size. 

        Args:
            dat           : [samples x channels] shape array of neural data 
            raw_threshold : [channels] shape vector of thresholds before multiplying. 
            mult          : threshold multiplier
            bin_size      : bin size in ms
            shift_size    : bin shift size in ms
            fs            : sampling frequency of neural data 

        Returns:
            crossings : [bins x channels] shape array of bin counts
        """
        
        # convert units to number of samples in bin, since bin_size is in millisecs
        bin_cnt = int(bin_size * fs / 1000)
        shift_cnt = int(shift_size * fs / 1000)
        
        # set data format and variables for numba
        dat = dat.astype('float32')
        n_bins = int(np.ceil(dat.shape[0] / shift_cnt))
        n_channels = dat.shape[1]
        crossings = np.zeros(shape=(n_bins, n_channels), dtype='float32')
        if raw_threshold is not None: 
            raw_threshold = raw_threshold.astype('float32')
        
        # compare threshold values with neural recordings
        threshold = raw_threshold * mult 
        crossings_total = dat <= threshold
        
        # iterate over each bin and count crossings
        for bin_i, i in enumerate(range(0, dat.shape[0], shift_cnt)): 
            crossings_bin = crossings_total[i:i+bin_cnt] 
            for ch in range(n_channels): 
                crossings[bin_i, ch] = np.sum(crossings_bin[:,ch])
        
        return crossings 
    
        
    def __call__(self, dat, fs, **kwargs): # the wrapper function
        """
        Counts threshold crossings for input neural data. 

        Args:
            dat           : [samples x channels] shape array of neural data 
            raw_threshold : [channels] shape vector of thresholds before multiplying, if already
                            calculated. included to avoid redundant calculations. 

        Returns:
            out : dict of threshold crossing counts, each following [bins x channels] shape
        """
        out = {} 
        
        self.fs = fs
        # compute raw threshold value to avoid redundant calculations
        if self.raw_thresholds is None:
            self.raw_thresholds = self.get_raw_threshold(dat, self.method)
        
        # for each multiplier, get the threshold crossing bin counts
        for mult in self.multipliers: 
            out[f"threshold_{mult}"] = self.count_threshold_crossings(dat, self.raw_thresholds, mult, 
                                                                      self.bin_size, self.shift_size, self.fs)
        
        return out 
    
    
    def summary(self, table_width=60): 
        """
        Provides a summary of the SpikePowExtractor class object. 
        Args:
            table_width : width of summary table. Defaults to 60 and minimum 50.
        """
        # logic for setting table width and checking whether parameters are valid
        if table_width < 50: 
            raise ValueError("Minimum width is 50")
        max_width = table_width - 2
        
        # prints table header
        print("┌" + "─" * max_width + "┐")
        title = "Threshold crossing counter configuration"
        left = int(np.ceil((max_width - len(title)) / 2))
        right = int(np.floor((max_width - len(title)) / 2))
        print("│" + " " * left + title + " " * right + "│")
        print("├" + "─" * max_width + "┤")
        
        # Handle potentially long text for threshold multipliers
        coltext = "│ Threshold multipliers   : " + " |"
        space_width = table_width - len(coltext)
        mult_text = chunk_text(str(self.multipliers), max_width=space_width)
        
        # print threshold multipliers line by line
        for i, line in enumerate(mult_text): 
            if i == 0 : 
                print_line(f"│ Threshold multipliers   : {line}", max_width)
            else: 
                print_line(f"│                           {line}", max_width)
        
        # print other parameters
        print_line(f"│ Signal sampling freq    : {self.fs} Hz ", max_width)
        print_line(f"│ Bin size                : {self.bin_size} ms", max_width)
        print_line(f"│ Shift size              : {self.shift_size} ms", max_width)
        print("└" + "─" * max_width + "┘")
    

class SpikePowExtractor: 
    def __init__(self, freq_bands, bin_size, shift_size, ord=4): 
        """
        Initializes the SpikePowExtractor class used for extracting spiking bandpower for a
        specified frequency band. Used for offline sliding window analysis. 
        
        Args:
            freq_bands : list of frequency bands.
            bin_size   : size of the bin in ms.
            shift_size : size of the shift between bins in ms.
            fs         : sampling frequency of neural data. 
            ord        : optional parameter for order of butterworth filter. defaults to 4.
        """
        self.freq_bands = freq_bands 
        self.ord = ord
        self.bin_size = bin_size 
        self.shift_size = shift_size 
        self.fs = 0
        
        # handle edge case for flexibility: input is a single frequency band
        if not isinstance(freq_bands, list): 
            self.freq_bands = [freq_bands]
    
    
    def add_band(self, band):
        """Adds new frequency band to extractor configuration"""
        if band not in self.freq_bands: 
            self.freq_bands.append(band) 
        
        
    def remove_band(self, band): 
        """Removes frequency band from extractor configuration"""
        if band in self.freq_bands: 
            self.freq_bands.remove(band)
        
        
    def _get_band_dat(self, dat, band, fs, ord=4): 
        """Bandpass filters neural data for given frequency band"""
        if self.ord != ord: 
            ord = self.ord
        dat_band = ButterworthFilter("bandpass", Wn=band, fs=self.fs, ord=ord)(dat, fs=fs)
        return dat_band
    
    
    @staticmethod 
    @nb.jit(nopython=True)
    def _bin_bandpower(dat, bin_size, shift_size, fs): 
        """
        Calculates binned power for given signal. 

        Args:
            dat        : [samples x channels] shape array of neural data 
            bin_size   : size of the bin in ms
            shift_size : size of the shift between bins in ms
            fs         : sampling frequency of neural data

        Returns:
            binned_pow : [bins x channels] shaped power of signal for each time bin
        """
        # convert milliseconds to number of time samples, since input is in ms
        bin_cnt = int(bin_size * fs / 1000) 
        shift_cnt = int(shift_size * fs / 1000)  
        
        # calculates the power of signal for every timestep
        dat = dat.astype('float32')
        dat_pow = np.square(dat)
        
        # initialize band power variable
        n_bins = int(np.ceil(dat_pow.shape[0] / shift_cnt))
        n_channels = dat_pow.shape[1]
        binned_bandpow = np.zeros(shape=(n_bins, n_channels), dtype='float64')
        
        # iterate over each bin and channel to compute the bin average power
        for bin_i, i in enumerate(range(0, dat_pow.shape[0], shift_cnt)): 
            dat_bin = dat_pow[i:i+bin_cnt]
            for ch in range(n_channels): 
                binned_bandpow[bin_i, ch] = np.mean(dat_bin[:,ch])
                
        return binned_bandpow
    
    
    def __call__(self, dat, fs): 
        """
        Calculates binned power for signal within given frequency band. 

        Args:
            dat : [samples x channels] shape array of neural data 
            fs  : sampling frequency of neural data
        Returns:
            out : dictionary of spiking bandpower for each frequency band, each with shape 
                  [bins x channels]
        """
        out = {} 
        self.fs = fs
        
        # iterate over each frequency band 
        for band in self.freq_bands: 
            # get bandpassed neural data for frequency band
            if band == 'full': 
                band_dat = dat.copy() 
                name = "bandpower_full"
            else: 
                band_dat = self._get_band_dat(dat, band=band, fs=fs)
                name = f"bandpower_{band[0]}_{band[1]}Hz"
                
            # get bandpower for bandpassed neural data 
            binned_bandpow = self._bin_bandpower(band_dat, self.bin_size, self.shift_size, self.fs)
            out[name] = binned_bandpow
            
        return out


    def summary(self, table_width=60): 
        """
        Provides a summary of the SpikePowExtractor class object. 
        Args:
            table_width : width of summary table. Defaults to 60 and minimum 50.
        """
        # logic for setting table width and checking whether parameters are valid
        if table_width < 50: 
            raise ValueError("Minimum width is 50")
        max_width = table_width - 2
        
        # prints table header
        print("┌" + "─" * max_width + "┐")
        title = "Spike bandpower extractor configuration"
        left = int(np.ceil((max_width - len(title)) / 2))
        right = int(np.floor((max_width - len(title)) / 2))
        print("│" + " " * left + title + " " * right + "│")
        print("├" + "─" * max_width + "┤")
        
        # handle potentially long text for frequency bands
        coltext = "│ Frequency bands (Hz)    : " + " |"
        space_width = max_width - len(coltext) + 2
        freq_band_text = chunk_text(str(self.freq_bands), max_width=space_width)
        
        # print frequency bands line by line
        for i, line in enumerate(freq_band_text): 
            if i == 0 : 
                print_line(f"│ Frequency bands (Hz)    : {line}", max_width)
            else: 
                print_line(f"│                           {line}", max_width)

        print_line(f"│ Signal sampling freq    : {self.fs} Hz ", max_width)
        print_line(f"│ Bandpass filter         : {self.ord}th order butterworth", max_width)
        
        bin_samples = int(self.bin_size * self.fs / 1000)
        print_line(f"│ Bin size                : {self.bin_size} ms ({bin_samples} samples)", max_width)
        
        shift_samples = int(self.shift_size * self.fs / 1000)
        print_line(f"│ Shift size              : {self.shift_size} ms ({shift_samples} samples)", max_width)
        print("└" + "─" * max_width + "┘")



# -------------------------------------------------------------------------------------------------
# Helper functions for the classes
# -------------------------------------------------------------------------------------------------

def print_line(text, max_width): 
    """Calculates required amount of whitespace for max_width length text and fills in whitespace"""
    print(f"{text}" + " " * (max_width - len(text) + 1) + "│")


def chunk_text(text, max_width): 
    """Chunks text into max_width length lines"""
    return [text[i:i + max_width] for i in range(0, len(text), max_width)]


@nb.jit(nopython=True) 
def lstsq_pseudoinverse(X, y):
    """optimized function for calculating the Moore-Penrose pseudoinverse, given by A+ = (AT A)-1 AT"""
    W = np.linalg.solve(X.T.dot(X), X.T.dot(y)) 
    return W