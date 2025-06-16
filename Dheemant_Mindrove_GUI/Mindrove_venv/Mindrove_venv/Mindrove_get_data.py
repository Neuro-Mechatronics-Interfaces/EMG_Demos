from collections import deque
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from Filter_data import SignalProcessor

# TODO: Change `self.buffers` from list of `deque` to `np.array` with proper circular buffer index/slicing logic, then
#           Check with AI to see if it thinks numpy would speed up vs list/deque approach.
#           Consider asking GPT or Gemini for CircularBuffer class using numpy for the implementation.
# TODO: Look into how to use `numba` and `jit` to speed up any filtering/processing function where it iterates over channels to apply filters in a loop. 
#           May not work well with scipy filters, so this is a "stretch" goal, but can improve performance.

class MindRoveSensor:
    """Real-time EMG sensor integration using MindRove board."""
    def __init__(self, buffer_size=1000, num_channels=8, do_baseline= False, do_mvc = False, do_Best_Channel = False):
        """
        Initialize the sensor with buffers and parameters.
        
        Args:
            buffer_size (int): Maximum size of the circular buffer for each channel.
            num_channels (int): Number of EMG channels.
        """
        self.buffer_size = buffer_size
        self.num_channels = num_channels

        self.len_baseline_data = 10000
        self.len_mvc_data = 10000
        self.buffers = [deque(maxlen=buffer_size) for _ in range(num_channels)]
        self.buffers_stream = [deque(maxlen=2500) for _ in range(num_channels)]
        self.baseline = [deque(maxlen=self.len_baseline_data) for _ in range(num_channels)]
        self.mvc = [deque(maxlen=self.len_mvc_data) for _ in range(num_channels)]
        self.discarded_elements = [[] for _ in range(num_channels)]  # Store discarded elements
        
        # Initialize the MindRove board
        self.board_shim = None
        self.params = MindRoveInputParams()
        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.is_streaming = False
        self.collect_data = False

        # Initialize board-related properties
        self.emg_channels = None
        self.accel_channels = None
        self.sampling_rate = None
        self.gyro_channel_scalar = None 
        self.alpha = None
        self.beta = None
        self.wrist_orientation_estimate_gain = None
        self.orientation = np.zeros(3, dtype=np.float32)
        self.wrist_orientation = np.zeros(3, dtype=np.float32)
        
        self.do_baseline = do_baseline
        self.do_mvc = do_mvc
        self.do_Best_Channel = do_Best_Channel
        

    def start(self):
        """
        Start collecting EMG data in real-time and populate buffers.
        """
        if self.board_shim is None:
            try:
                # Initialize the board if not already initialized
                self.board_shim = BoardShim(self.board_id, self.params)
                self.board_shim.prepare_session()
                self.board_shim.start_stream()
                self.is_streaming = True

                # Set board-related properties
                self.emg_channels = BoardShim.get_emg_channels(self.board_id)
                self.accel_channels = BoardShim.get_accel_channels(self.board_id)
                self.gyro_channels = BoardShim.get_gyro_channels(self.board_id)
                self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
                self.gyro_channel_scalar = 0.00025 * (1/self.sampling_rate) # = 0.125 @ 500 Hz
                self.wrist_orientation_estimate_gain = 1.0


                self.alpha = 0.98 # gain on gyro contribution to orientation; accelerometer contribution is 1-_alpha.
                self.beta = 0.020
                print("Board started successfully.")

            except Exception as e:
                print(f"Error starting board: {e}")
                return

        window_size = 2  # seconds
        num_points = window_size * self.sampling_rate

        try:
            while self.is_streaming:
                if self.board_shim.get_board_data_count() > 0:
                    data = self.board_shim.get_board_data(num_points)
                    emg_data = data[self.emg_channels]  # Shape: (num_channels, num_samples)
                    accel_data = data[self.accel_channels] # Shape: (num_channels, num_samples)
                    gyro_data = data[self.gyro_channels] # Shape: (num_channels, num_samples)

                    # --- for EMG data ---
                    for channel in range(self.num_channels):
                        new_data = emg_data[channel]                        
                        # Handle baseline collection
                        if self.do_baseline and not hasattr(self, 'collecting_baseline'):
                            self.collecting_baseline = True  # Start tracking
                            self.baseline[channel].clear()  # Ensure fresh data
                            print(f"Starting baseline collection for Channel {channel}")

                        if self.do_baseline and self.collecting_baseline:
                            remaining = self.len_baseline_data - len(self.baseline[channel])
                            self.baseline[channel].extend(new_data[:remaining])  # Only take needed samples
                            if len(self.baseline[channel]) >= self.len_baseline_data:
                                print(f"Baseline data collected for Channel {channel}")
                                self.do_baseline = False  # Stop collecting
                                del self.collecting_baseline  # Reset flag

                        # Handle MVC collection
                        if self.do_mvc and not hasattr(self, 'collecting_mvc'):
                            self.collecting_mvc = True  # Start tracking
                            self.mvc[channel].clear()  # Ensure fresh data
                            print(f"Starting MVC collection for Channel {channel}")

                        if self.do_mvc and self.collecting_mvc:
                            remaining = self.len_mvc_data - len(self.mvc[channel])
                            self.mvc[channel].extend(new_data[:remaining])  # Only take needed samples
                            if len(self.mvc[channel]) >= self.len_mvc_data:
                                print(f"MVC data collected for Channel {channel}")
                                self.do_mvc = False  # Stop collecting
                                del self.collecting_mvc  # Reset flag

                        # Only store in current_buffer if data collection is active
                        # TODO: Strongly recommend changing this to a single CircularBuffer here, it should be as large as the
                        #           longest buffer that you require. Then, you would simply slice the "most-recent" part of the 
                        #           buffer to return whatever most-recent samples you want even if that's a subset of CircularBuffer.
                        #           
                        #   i.e. you should have buffer.get_recent_n_samples(n) so you can always pull the `n` most-recent samples.
                        #       -> Inside CircularBuffer there should be an indexing operation any time you add to your buffer so that
                        #           it tracks and handles returning those samples in the correct order.
                        if self.collect_data:
                            excess = max(0, len(self.buffers[channel]) + len(new_data) - self.buffer_size)
                            self.buffers_stream[channel].extend(new_data)
                            if excess > 0:
                                discarded = list(self.buffers[channel])[:excess]
                                self.discarded_elements[channel].extend(discarded)
                            self.buffers[channel].extend(new_data)


                    # # --- for Accel & Gyro data ---
                    # for chan in range(3):
                    acc_data = np.mean(accel_data[:, :], axis=1) # Extract accelerometer channels (shape: 3 x num_samples)
                    gyro_contribution = np.sum(gyro_data[:, :] * self.gyro_channel_scalar, axis=1)

                    acc_contribution = np.zeros(3)
                    acc_contribution[0] = np.atan2(acc_data[1], acc_data[2])
                    acc_contribution[1] = np.atan2(-acc_data[0], np.sqrt(acc_data[1]**2 + acc_data[2]**2))

                    # Update orientation at wristband:
                    self.orientation = self.alpha * (gyro_contribution + self.orientation) + self.beta * acc_contribution

                    # Based on wristband rotation, update orientation at the wrist:
                    self.wrist_orientation = self.alpha * (gyro_contribution * self.wrist_orientation_estimate_gain + self.wrist_orientation) + self.beta * acc_contribution * self.wrist_orientation_estimate_gain
        

                    time.sleep(0.01)   # Simulate a 10ms data arrival interval

        except KeyboardInterrupt:
            print("Data collection stopped manually.")
        except Exception as e:
            print(f"Error during data collection: {e}")
        finally:
            print("Exiting sensor.start() loop.")
            # self.stop()


    def get_wrist_orientation(self):
        """
        Returns the current wrist orientation estimate.

        Returns:
            np.ndarray: A 3-element array representing orientation (e.g., roll, pitch, yaw).
        """
        return self.wrist_orientation
    
    def get_buffer(self, channel):
        """
        Get the current buffer for a specific channel.

        Args:
            channel (int): Channel index.

        Returns:
            deque: Current buffer content for the given channel.
        """
        return self.buffers[channel]
    
    def get_buffers_stream (self, channel):

        return self.buffers_stream[channel]

    def get_discarded_elements(self):
        """
        Retrieve all discarded elements for all channels as a DataFrame.

        Returns:
            pd.DataFrame: Discarded elements organized by channels.
        """
        df = pd.DataFrame(self.discarded_elements).transpose()
        print("Discarded Elements is being called>>")
        return df

    def get_NRO_elements(self, num_channels=8, fs=500): # Not Online, Just Offline Data Collection
        filtered_signals = []
        rms_values = []
        processor = SignalProcessor(self, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        df_data = pd.DataFrame(self.discarded_elements).transpose()
        unfiltered_data = pd.DataFrame(self.discarded_elements).transpose()
        unfiltered_data.columns = [f"Channel {i+1}" for i in range(num_channels)]

        for i in range(num_channels):
            channel_data = df_data.iloc[:, i].values
            filtered_data = processor.preprocess_filters(channel_data)
            filtered_data = processor.moving_window_rms(filtered_data, 200)
            rms_data = processor.calculate_rms(filtered_data)
            # normalized_data = np.clip(normalized_data, 0, 1)  # Ensure values are within [0,1]
            filtered_signals.append(filtered_data)
            rms_values.append(rms_data)
        
        filtered_signals = pd.DataFrame(filtered_signals).T
        filtered_signals.columns = [f"Channel {i+1}" for i in range(num_channels)]
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 12), sharex=True)
        fig.suptitle("Filtered EMG Data", fontsize=16)

        for i in range(num_channels):
            axes[i].plot(filtered_signals.iloc[:, i], label=f'Channel {i + 1}', linewidth=0.7)
            axes[i].set_title(f'Channel {i + 1}')
            axes[i].set_ylabel("EMG_Amp")
            axes[i].grid(True)
            axes[i].legend()
            axes[i].relim()
            axes[i].autoscale_view()

        axes[-1].set_xlabel("Time Points")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


        return unfiltered_data, filtered_signals, rms_values

    def clear_discarded_elements(self):
        """Clear all discarded elements for all channels."""
        self.discarded_elements = [[] for _ in range(len(self.discarded_elements))]  # Reset to empty lists
        self.buffers = [deque(maxlen=self.buffer_size) for _ in range(self.num_channels)]  # Reset buffers
        print("Discarded elements have been cleared.")

    def stop(self):
        """
        Stop the EMG data stream and release resources.
        """
        if self.board_shim is not None and self.is_streaming:
            try:
                self.board_shim.stop_stream()
                self.board_shim.release_session()
                self.board_shim = None
                self.is_streaming = False
                print("Sensor session stopped and resources released.")
            except Exception as e:
                print(f"Error stopping board: {e}")

