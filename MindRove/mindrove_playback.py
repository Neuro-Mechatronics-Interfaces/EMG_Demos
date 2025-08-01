import time
import threading
import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
from collections import deque


class MindRovePlaybackClient:
    def __init__(self, filepath, delimiter='\t', block_size=50, loopback=False, enable_lsl=False, sampling_rate=1000, verbose=False):
        self.filepath = filepath
        self.block_size = block_size
        self.loopback = loopback
        self.enable_lsl = enable_lsl
        self.verbose = verbose
        self.sampling_rate = sampling_rate  # MindRove default (can be overwritten)

        # === Load CSV data ===
        df = pd.read_csv(filepath, delimiter=delimiter)

        # EMG data exists in only the first 8 columns
        if df.shape[1] > 8:
            df = df.iloc[:, :8]
        self.ch_names = list(df.columns)
        self.data = df.values.T.astype(np.float32)  # shape (n_channels, n_samples)

        self.n_channels, self.n_samples = self.data.shape
        self.current_index = 0
        self.total_samples = 0
        self.streaming = False
        self.buffer = np.zeros_like(self.data)
        self.thread = None
        self.lock = threading.Lock()

        if self.enable_lsl:
            self._initialize_lsl_stream()

    def _initialize_lsl_stream(self):
        info = StreamInfo(
            name='MindRoveData',
            type='EMG',
            channel_count=self.n_channels,
            nominal_srate=self.sampling_rate,
            channel_format='float32',
            source_id='MindRovePlaybackClient'
        )
        chns = info.desc().append_child('channels')
        for ch_name in self.ch_names:
            chns.append_child('channel').append_child_value('label', ch_name)
        self.lsl_outlet = StreamOutlet(info)

    def start_streaming(self):
        if not self.streaming:
            self.streaming = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            if self.verbose:
                print("MindRove streaming started...")

    def stop_streaming(self):
        self.streaming = False
        if self.thread:
            self.thread.join()
            self.thread = None
            if self.verbose:
                print("MindRove streaming stopped.")

    def _stream_loop(self):
        interval = self.block_size / self.sampling_rate
        while self.streaming:
            with self.lock:
                end = min(self.current_index + self.block_size, self.n_samples)
                chunk = self.data[:, self.current_index:end]
                self.buffer[:, self.total_samples:end] = chunk
                self.total_samples += chunk.shape[1]
                self.current_index = end

                if self.enable_lsl and self.lsl_outlet:
                    self.lsl_outlet.push_chunk(chunk.T.tolist())

                if self.verbose:
                    print(f"Streaming index: {self.current_index}/{self.n_samples}")

                if self.current_index >= self.n_samples:
                    if self.loopback:
                        if self.verbose:
                            print("Looping playback...")
                        self.current_index = 0
                        self.total_samples = 0
                        self.buffer.fill(0)
                    else:
                        break

            time.sleep(interval)

    def get_latest_window(self, window_ms):
        samples_per_window = int(window_ms / 1000 * self.sampling_rate)
        start = max(0, self.total_samples - samples_per_window)
        return self.buffer[:, start:self.total_samples]

    def close(self):
        self.stop_streaming()


class LSLClient:
    def __init__(self, maxlen=10000, stream_type="EMG"):
        print(f"[LSLClient] Looking for a stream of type '{stream_type}'...")
        streams = resolve_byprop("type", stream_type, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream with type '{stream_type}' found.")

        self.inlet = StreamInlet(streams[0])
        self.stream_info = self.inlet.info()
        self.n_channels = self.stream_info.channel_count()
        self.fs = self._get_sampling_rate()
        self.channel_labels, self.units = self._get_channel_metadata()

        self.buffers = [deque(maxlen=maxlen) for _ in range(self.n_channels)]
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._pull_data_loop, daemon=True)
        self.thread.start()
        print(f"[LSLClient] Connected to stream '{self.stream_info.name()}'")
        print(f"  Channels: {self.n_channels}, Sampling Rate: {self.fs} Hz")

    def _get_sampling_rate(self):
        try:
            rate = self.stream_info.nominal_srate()
            return float(rate) if rate > 0 else None
        except Exception:
            return None

    def _get_channel_metadata(self):
        try:
            ch_info = self.stream_info.desc().child("channels").child("channel")
            labels = []
            units = []
            for _ in range(self.n_channels):
                labels.append(ch_info.child_value("label") or f"Ch{_}")
                units.append(ch_info.child_value("unit") or "unknown")
                ch_info = ch_info.next_sibling()
            return labels, units
        except Exception:
            return [f"Ch{i}" for i in range(self.n_channels)], ["unknown"] * self.n_channels

    def _pull_data_loop(self):
        while self.running:
            sample, _ = self.inlet.pull_sample(timeout=0.1)
            if sample is not None:
                with self.lock:
                    for ch, val in enumerate(sample):
                        if ch < self.n_channels:
                            self.buffers[ch].append(val)

    def get_samples(self, channel: int, n_samples: int):
        with self.lock:
            buf = list(self.buffers[channel])
        if len(buf) < n_samples:
            buf = [0.0] * (n_samples - len(buf)) + buf
        return buf[-n_samples:]

    def stop(self):
        self.running = False
        self.thread.join()
        print("[LSLClient] Stopped.")