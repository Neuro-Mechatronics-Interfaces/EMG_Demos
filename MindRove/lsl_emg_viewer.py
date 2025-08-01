import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from mindrove_playback import LSLClient  # Replace with correct path if needed
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=10, highcut=400, fs=1000, order=4):
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)


class LSLEMGPlot:
    def __init__(self, client, n_channels=4, duration_sec=4, refresh_ms=50, y_range=(-5000, 5000)):
        self.client = client
        self.fs = int(self.client.fs)

        self.n_channels = min(n_channels, self.client.n_channels)
        self.buffer_size = int(duration_sec * self.fs)
        self.refresh_ms = refresh_ms

        self.plot_buffers = [np.zeros(self.buffer_size) for _ in range(self.n_channels)]
        self.time_axis = np.arange(self.buffer_size) / self.fs

        # === PyQtGraph GUI ===
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="LSL EMG Viewer")
        self.win.setWindowTitle("Real-Time LSL EMG Plot")

        self.raw_curves = []
        self.rms_curves = []
        for i in range(self.n_channels):
            p = self.win.addPlot(title=f"Channel {i}")
            p.setLabel('left', "µV")
            p.setYRange(*y_range)
            p.enableAutoRange('y', True)

            raw = p.plot(pen=pg.mkPen('blue', width=1))
            rms = p.plot(pen=pg.mkPen('orange', width=2))

            self.raw_curves.append(raw)
            self.rms_curves.append(rms)
            self.win.nextRow()

        self.win.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.refresh_ms)

        self.app.aboutToQuit.connect(self.cleanup)

    def compute_rms(self, signal, window_size=20):
        """
        Compute sliding RMS with specified window size (in samples).
        """
        if len(signal) < window_size:
            return np.zeros_like(signal)
        squared = np.square(signal)
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared, window, mode='valid'))
        # Pad to match the original length
        pad = (len(signal) - len(rms)) // 2
        return np.pad(rms, (pad, len(signal) - len(rms) - pad), mode='edge')

    def update(self):
        """
        Pull new LSL data and update plots.
        """
        for i in range(self.n_channels):
            new_data = self.client.get_samples(i, self.buffer_size)
            new_data = np.array(new_data)

            # Detrend data
            new_data -= np.mean(new_data)

            # Impleent bandpass between 10Hz and 400Hz
            new_data = bandpass_filter(new_data, 10, 400, fs=self.fs)

            self.plot_buffers[i] = np.roll(self.plot_buffers[i], -len(new_data))
            self.plot_buffers[i][-len(new_data):] = new_data[-len(self.plot_buffers[i]):]

            self.raw_curves[i].setData(self.time_axis, self.plot_buffers[i])
            self.rms_curves[i].setData(self.time_axis, self.compute_rms(self.plot_buffers[i]))

    def cleanup(self):
        print("Stopping LSL stream...")
        self.client.stop()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    client = LSLClient(stream_type="EMG")
    viewer = LSLEMGPlot(client=client, n_channels=4, duration_sec=4, refresh_ms=50, y_range=(-4000, 4000))
    viewer.run()
