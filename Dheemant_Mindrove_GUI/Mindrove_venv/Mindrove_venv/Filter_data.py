from scipy.signal import butter, filtfilt, iirnotch, lfilter, lfilter_zi
import numpy as np
from collections import deque

class SignalProcessor:
    """Class for applying signal processing techniques to EEG data."""

    def __init__(self, sensor, fs, highpass_cutoff=10.0, notch_freq=60.0, notch_q=30, highpass_order=4):
        """
        Initialize the signal processor with filters and settings.

        Args:
            sensor: Sensor object providing EEG data.
            fs (float): Sampling frequency of the signal.
            highpass_cutoff (float): High-pass filter cutoff frequency (Hz).
            notch_freq (float): Notch filter target frequency (Hz).
            notch_q (float): Quality factor for the notch filter.
            highpass_order (int): Order of the high-pass filter.
        """
        self.sensor = sensor
        self.fs = fs
        self.highpass_cutoff = highpass_cutoff
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.highpass_order = highpass_order
        self.total_samples_processed = 0

        # Design filters
        self.b_high, self.a_high, self.zi_high = self._design_highpass_filter()
        self.b_band, self.a_band, self.zi_band = self._design_butter_bandpass_filter()
        self.b_notch, self.a_notch, self.zi_notch = self._design_notch_filter()
        self.zi_high_stream = self.zi_high
        self.zi_band_stream = self.zi_band
        self.zi_notch_stream = self.zi_notch

    def _design_notch_filter(self):
        """Design a notch filter to remove a specific frequency."""
        b, a = iirnotch(self.notch_freq, self.notch_q, self.fs)
        zi = lfilter_zi(b, a)
        return b, a, zi

    def _design_highpass_filter(self):
        """Design a high-pass filter using a Butterworth filter."""
        nyquist = 0.5 * self.fs
        normal_cutoff = self.highpass_cutoff / nyquist
        b, a = butter(self.highpass_order, normal_cutoff, btype='high', analog=False)
        zi = lfilter_zi(b, a)
        return b, a, zi
    
    def _design_butter_bandpass_filter(self,lowcut = 8, highcut = 248, fs= 500, order=4):
        """Design a band-pass filter using a Butterworth filter."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        zi = lfilter_zi(b, a)
        return b, a, zi

    def rectify_emg(self, data):
        """Rectifies EMG data by converting all values to their absolute values."""
        return np.abs(data)

    def moving_window_rms(self, signal, window_size):
        """Compute windowed RMS of the signal."""
        return np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode='same'))

    def non_ovlp_calculate_rms(self, data, window_size):
        """Calculate RMS features for each channel using non-overlapping windows.

        Args:
            data (numpy.ndarray): EEG data of shape (channels, samples).
            window_size (int): Size of each window (in samples).

        Returns:
            numpy.ndarray: RMS features of shape (channels, windows).
        """
        n_channels, n_samples = data.shape
        n_windows = n_samples // window_size
        rms_features = np.zeros((n_channels, n_windows))

        for ch in range(n_channels):
            for i in range(n_windows):
                window = data[ch, i * window_size:(i + 1) * window_size]
                rms_features[ch, i] = np.sqrt(np.mean(window ** 2))

        return rms_features

    def calculate_rms(self, window):
        """Calculate RMS value for a single data window."""
        return np.sqrt(np.mean(np.array(window) ** 2))

    def preprocess_filters(self, data):
        """Preprocess the data by applying filters and rectification.

        Args:
            data (numpy.ndarray): Raw signal data.

        Returns:
            numpy.ndarray: Filtered and rectified data.
        """
        data = np.array(data)  # Ensure input is a NumPy array

        # Apply band-pass filter
        data = filtfilt(self.b_band, self.a_band, data)
        # Apply notch filter
        data = filtfilt(self.b_notch, self.a_notch, data)

        # Apply band-pass filter
        # data, self.zi_band = lfilter(self.b_band, self.a_band, data, zi=self.zi_band)
        # Apply notch filter
        # data, self.zi_notch = lfilter(self.b_notch, self.a_notch, data, zi=self.zi_notch)
        return data
    
    
    def preprocess_filters_stream(self, data):
        data = np.array(data)  # Ensure input is a NumPy array

        # Apply band-pass filter
        # data = filtfilt(self.b_high, self.a_high, data)
        # data, self.zi_high_stream = lfilter(self.b_high, self.a_high, data, zi=self.zi_high_stream)
        # data = filtfilt(self.b_band, self.a_band, data)
        data = filtfilt(self.b_band, self.a_band, data)
        # Apply notch filter
        data = filtfilt(self.b_notch, self.a_notch, data)
        return data

    # TODO: Consider changing this instead of returning arrays, assign to attribute buffers here.
    #       -> You would then only have the end-result here self.filtered_buffer
    def preproc_loop_buffer(self, window_size):
        """Preprocess EMG data buffers in real-time.

        Returns:
            Tuple: (smoothed_buffers, preprocessed_buffers)
        """
        pre_rectification_buffers = [[] for _ in range(self.sensor.num_channels)]
        preprocessed_buffers = [[] for _ in range(self.sensor.num_channels)]
        preprocessed_buffers_stream = [[] for _ in range(self.sensor.num_channels)]

        smoothed_buffers = [[] for _ in range(self.sensor.num_channels)]
        smoothed_buffers_stream =  [[] for _ in range(self.sensor.num_channels)]

        num_samples_in_buffer = len(self.sensor.get_buffer(0))

        if self.total_samples_processed < 300:
            # Calculate how many of the *new* samples need to be discarded.
            # `total_samples_processed` is the count of samples from *before* this processing cycle.
            samples_to_chop = 300 - self.total_samples_processed
        else:
            # We are past the artifact zone, so we don't need to chop anything.
            samples_to_chop = 0
        
        # Update the total count with the new samples we are about to process.
        # We do this now so the count is correct for the next cycle.
        self.total_samples_processed += num_samples_in_buffer

        for channel in range(self.sensor.num_channels):
            # Step 1: Get raw data
            raw_buffer = list(self.sensor.get_buffer(channel))
            raw_buffers_stream = list(self.sensor.get_buffers_stream(channel))

            # Step 2: Bandpass + Notch + Rectification
            pre_rectification_buffer = self.preprocess_filters(raw_buffer)
            pre_rectification_buffer_ = pre_rectification_buffer[samples_to_chop:]
            preprocessed_buffer = self.rectify_emg(pre_rectification_buffer_)
            preprocessed_buffer_stream = self.preprocess_filters_stream(raw_buffers_stream)

            pre_rectification_buffers[channel] = pre_rectification_buffer_
            preprocessed_buffers[channel] = preprocessed_buffer
            preprocessed_buffers_stream[channel] = preprocessed_buffer_stream

            # Step 3: RMS smoothing
            smoothed_buffer = self.moving_window_rms(np.array(preprocessed_buffer), window_size)
            smoothed_buffer_stream = self.moving_window_rms(np.array(self.rectify_emg(preprocessed_buffer_stream)), window_size)

            smoothed_buffers[channel] = smoothed_buffer
            smoothed_buffers_stream[channel] = smoothed_buffer_stream
            # self.filtered_buffer[channel] = smoothed_buffer

        return smoothed_buffers, preprocessed_buffers, smoothed_buffers_stream, pre_rectification_buffers, preprocessed_buffers_stream

    
    
    
    def preproc_loop_discard_elem(self, window_size):
        """Preprocess EMG data buffers in real-time.

        Args:
            window_size (int): RMS window size (in samples).

        Returns:
            list of numpy.ndarray: Smoothed signal buffers for each channel.
        """
        preprocessed_buffers = [[] for _ in range(self.sensor.num_channels)]
        smoothed_buffers = [[] for _ in range(self.sensor.num_channels)]
        raw_buffer = list(self.sensor.get_discarded_elements())

        for channel in range(self.sensor.num_channels):
            # Get raw buffer from the sensor
            
            # raw_buffer = np.array(list(self.sensor.get_buffer(channel)))

            # Apply preprocessing (filters and rectification)
            preprocessed_buffer = self.preprocess_filters(raw_buffer)
            preprocessed_buffers[channel] = preprocessed_buffer

            # Smooth the signal using RMS
            smoothed_buffer = self.moving_window_rms(np.array(preprocessed_buffer), window_size)
            smoothed_buffers[channel] = smoothed_buffer

        return smoothed_buffers
