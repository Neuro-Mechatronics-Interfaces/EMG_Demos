# fake_mindrove.py
import numpy as np
#

class FakeMindrove:
    def __init__(self, board_id, input_params):
        self.board_id = board_id
        self.input_params = input_params
        self.is_streaming = False
        self.is_session_prepared = False
        self.sampling_rate = 500  # Hz
        self.num_channels = 8
        self.data_buffer = np.empty((self.num_channels, 0))
        self.rng = np.random.default_rng()

    def prepare_session(self):
        self.is_session_prepared = True
        print("FakeMindrove session prepared.")

    def start_stream(self, num_samples=450000, streamer_params=None):
        self.is_streaming = True
        print("FakeMindrove stream started.")

    def stop_stream(self):
        self.is_streaming = False
        print("FakeMindrove stream stopped.")

    def release_session(self):
        self.is_session_prepared = False
        print("FakeMindrove session released.")

    def get_sampling_rate(self, board_id):
        return self.sampling_rate

    def get_exg_channels(self, board_id):
        return list(range(self.num_channels))

    def get_board_data(self, num_samples=None, preset=None):
        if not self.is_streaming or not self.is_session_prepared:
            return np.zeros((self.num_channels, 0))

        samples = num_samples if num_samples else self.sampling_rate
        data = []

        for ch in range(self.num_channels):
            # Baseline EMG: band-limited noise
            baseline = self.rng.normal(loc=0, scale=20, size=samples)

            # Random EMG burst
            if self.rng.random() < 0.05:
                burst_duration = self.rng.integers(100, 500)
                burst_start = self.rng.integers(0, samples - burst_duration)
                baseline[burst_start:burst_start + burst_duration] += self.rng.normal(loc=100, scale=30,
                                                                                      size=burst_duration)

            # Occasional motion artifact
            if self.rng.random() < 0.02:
                freq = self.rng.uniform(1, 3)
                amplitude = self.rng.uniform(50, 150)
                t = np.linspace(0, samples / self.sampling_rate, samples)
                baseline += amplitude * np.sin(2 * np.pi * freq * t)

            data.append(baseline)

        data_array = np.array(data)
        self.data_buffer = np.hstack((self.data_buffer, data_array))

        if num_samples is None:
            data_to_return = self.data_buffer
            self.data_buffer = np.empty((self.num_channels, 0))
        else:
            data_to_return = self.data_buffer[:, :samples]
            self.data_buffer = self.data_buffer[:, samples:]
        return data_to_return

    def get_board_id(self):
        return self.board_id

    def is_prepared(self):
        return self.is_session_prepared

    def insert_marker(self, value, preset=None):
        print(f"Marker inserted: {value}")
