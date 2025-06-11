# demo_fake_mindrove_realtime.py
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations
from mindrove.board_shim import (
    # BoardShim,
    BoardIds,
    MindRoveInputParams,
)

from fake_mindrove import FakeMindrove as BoardShim

class ScrollingEMGPlot:
    def __init__(self, n_channels=4, duration_sec=4, refresh_ms=50, band=(10, 500), y_range=(-500, 500)):
        # === FakeMindrove Setup ===
        self.params = MindRoveInputParams()
        self.board_id = BoardIds.MINDROVE_WIFI_BOARD
        self.board = BoardShim(self.board_id, None)

        self.board.prepare_session()
        self.board.start_stream()

        self.sampling_rate = self.board.get_sampling_rate(self.board_id)
        print(f"Sampling rate: {self.sampling_rate} Hz")
        self.channel_indices = list(range(n_channels))
        self.n_channels = n_channels
        self.buffer_size = int(duration_sec * self.sampling_rate)
        self.refresh_ms = refresh_ms
        self.rms_window = int(0.050 * self.sampling_rate)  # 50 ms RMS window
        self.band = band

        # === Data Buffer ===
        self.plot_buffers = [np.zeros(self.buffer_size) for _ in range(self.n_channels)]

        # === PyQtGraph Setup ===
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Scrolling EMG Plot")
        self.win.setWindowTitle("FakeMindrove EMG Plot")

        self.plots = []
        self.raw_curves = []
        self.rms_curves = []
        for i in range(self.n_channels):
            p = self.win.addPlot(title=f"Electrode {self.channel_indices[i]}")
            p.setLabel('left', "µV")
            p.setYRange(*y_range)
            p.enableAutoRange('y', True)

            raw_curve = p.plot(pen=pg.mkPen(color='blue', width=1))
            rms_curve = p.plot(pen=pg.mkPen(color='orange', width=2))

            self.plots.append(p)
            self.raw_curves.append(raw_curve)
            self.rms_curves.append(rms_curve)
            self.win.nextRow()

        self.win.show()

        # === Timer Loop ===
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.refresh_ms)

        # Cleanup on close
        self.app.aboutToQuit.connect(self.cleanup)

    def compute_rms(self, signal):
        window_size = int(0.050 * self.sampling_rate)
        if len(signal) < window_size:
            return np.zeros_like(signal)
        squared = np.square(signal)
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared, window, mode='same'))
        return rms

    def update(self):
        data = self.board.get_board_data()
        if data.shape[1] > 0:
            rms_vector = []
            for i, ch in enumerate(self.channel_indices):
                raw = np.copy(data[ch])

                # Detrend and filter the data
                DataFilter.detrend(raw, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(raw, self.sampling_rate, self.band[0], self.band[1], 2, FilterTypes.BUTTERWORTH, 0)

                rms = self.compute_rms(raw)
                rms_vector.append(rms[-1])

                # Scroll and update plot data
                data_to_use = raw
                num_new = min(len(data_to_use), len(self.plot_buffers[i]))
                self.plot_buffers[i] = np.roll(self.plot_buffers[i], -num_new)
                self.plot_buffers[i][-num_new:] = data_to_use[-num_new:]

                # Update curves
                self.raw_curves[i].setData(self.plot_buffers[i])
                self.rms_curves[i].setData(self.compute_rms(self.plot_buffers[i]))

    def cleanup(self):
        print("Stopping stream and releasing resources.")
        self.board.stop_stream()
        self.board.release_session()

    def run(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    plotter = ScrollingEMGPlot(
        n_channels=4,
        duration_sec=4,
        band=(10, 240),
        y_range=(-500, 500)
    )
    plotter.run()
