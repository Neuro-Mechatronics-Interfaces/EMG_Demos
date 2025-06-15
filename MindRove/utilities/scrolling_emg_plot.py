# demo_fake_mindrove_realtime.py
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations
from mindrove.board_shim import (
    BoardShim,
    BoardIds,
    MindRoveInputParams,
)


class ScrollingEMGPlot:
    """
    A class to visualize EMG data in real-time using PyQtGraph.

    Parameters:
        n_channels (int): Number of EMG channels to visualize.
        duration_sec (int): Duration of data buffer in seconds.
        refresh_ms (int): Refresh rate in milliseconds for the plot.
        band (tuple): Frequency band for bandpass filtering (low, high).
        y_range (tuple): Y-axis range for the plot.

    """
    def __init__(self, board=None, board_id=None, n_channels=4, duration_sec=4, refresh_ms=50, band=(10, 500), y_range=(-500, 500)):

        # === FakeMindrove Setup ===
        self.params = MindRoveInputParams()

        if board is None and board_id is None:
            board_id = BoardIds.SYNTHETIC_BOARD
            board = BoardShim(board_id, self.params)
        if board is None and board_id is not None:
            self.params.master_board = board_id
            self.params.other_info = str(board_id).zfill(32)
            board = BoardShim(board_id, self.params)
        if board is not None and board_id is None:
            board_id = board.get_board_id()

        self.board_id = board_id
        self.board = board

        # Prepare session if not already
        if not self.board.is_prepared():
            self.board.prepare_session()

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
        self.time_axis = np.arange(self.buffer_size) / self.sampling_rate  # Time in seconds for x-axis

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
        """
        Compute the Root Mean Square (RMS) of a signal using a sliding window.

        Args:
            signal (np.ndarray): Input signal array.

        Returns:
            np.ndarray: RMS values computed over a sliding window.

        """
        window_size = int(0.050 * self.sampling_rate)
        if len(signal) < window_size:
            return np.zeros_like(signal)
        squared = np.square(signal)
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared, window, mode='same'))
        return rms

    def update(self):
        """
        Update the plot with new data from the board.

        """
        data = self.board.get_board_data()  # Get all the data from the buffer
        if data.shape[1] > 0:
            rms_vector = []
            for i, ch in enumerate(self.channel_indices):
                raw = np.copy(data[ch])

                # Detrend and filter the data
                DataFilter.detrend(raw, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(raw, self.sampling_rate, self.band[0], self.band[1], 2, FilterTypes.BUTTERWORTH, 0)

                # Compute RMS and update the RMS vector
                rms = self.compute_rms(raw)
                rms_vector.append(rms[-1])

                # Scroll and update plot data
                data_to_use = raw
                num_new = min(len(data_to_use), len(self.plot_buffers[i]))
                self.plot_buffers[i] = np.roll(self.plot_buffers[i], -num_new)
                self.plot_buffers[i][-num_new:] = data_to_use[-num_new:]

                # Update curves
                self.raw_curves[i].setData(self.time_axis, self.plot_buffers[i])
                self.rms_curves[i].setData(self.time_axis, self.compute_rms(self.plot_buffers[i]))

    def cleanup(self):
        """
        Cleanup function to stop the stream and release resources.

        """
        print("Stopping stream and releasing resources.")
        self.board.stop_stream()
        self.board.release_session()

    def run(self):
        """
        Start the application event loop.

        """
        self.board.start_stream()
        sys.exit(self.app.exec_())

