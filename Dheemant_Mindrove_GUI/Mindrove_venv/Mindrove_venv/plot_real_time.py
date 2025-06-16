# ------------------------------------------------------------------------------------------


# MINDROVE (PLOT_REAL_TIME.PY)_____________________________WORKING


# ------------------------------------------------------------------------------------------

import argparse
import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations


class Graph:
    def __init__(self, board_shim):
        # Set pyqtgraph options
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        # Create Qt application and main window
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title='Mindrove Plot')
        self.win.resize(800, 600)

        # Initialize plots
        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()

        # Start update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.app.exec()

    def _init_pens(self):
        self.pens = []
        self.brushes = []
        colors = ['#A54E4E', '#A473B6', '#5B45A4', '#2079D2', '#32B798', '#2FA537', '#9DA52F', '#A57E2F', '#A53B2F']
        for color in colors:
            self.pens.append(pg.mkPen({'color': color, 'width': 2}))
            self.brushes.append(pg.mkBrush(color))

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for i, channel in enumerate(self.exg_channels):
            plot = self.win.addPlot(row=i, col=0)
            plot.showAxis('left', False)
            plot.setMenuEnabled('left', False)
            plot.showAxis('bottom', False)
            plot.setMenuEnabled('bottom', False)
            if i == 0:
                plot.setTitle('Time Series Plot')
            self.plots.append(plot)
            curve = plot.plot(pen=self.pens[i % len(self.pens)])
            self.curves.append(curve)

    def _init_psd(self):
        self.psd_plot = self.win.addPlot(row=0, col=1, rowspan=len(self.exg_channels) // 2)
        self.psd_plot.setTitle('PSD Plot')
        self.psd_plot.setLogMode(False, True)
        self.psd_curves = []
        self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
        for i, channel in enumerate(self.exg_channels):
            psd_curve = self.psd_plot.plot(pen=self.pens[i % len(self.pens)])
            self.psd_curves.append(psd_curve)

    def _init_band_plot(self):
        self.band_plot = self.win.addPlot(row=len(self.exg_channels) // 2, col=1, rowspan=len(self.exg_channels) // 2)
        self.band_plot.setTitle('Band Power Plot')
        self.band_plot.showAxis('left', False)
        self.band_plot.showAxis('bottom', False)
        x = [1, 2, 3, 4, 5]
        y = [0, 0, 0, 0, 0]
        self.band_bar = pg.BarGraphItem(x=x, height=y, width=0.8, brush=self.brushes[0])
        self.band_plot.addItem(self.band_bar)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        avg_bands = [0, 0, 0, 0, 0]
        for i, channel in enumerate(self.exg_channels):
            # Apply filters and detrend
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 30.0, 56.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 2, FilterTypes.BUTTERWORTH.value, 0)

            # Update time series plot
            self.curves[i].setData(data[channel].tolist())

            if data.shape[1] > self.psd_size:
                # Update PSD plot
                psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2, self.sampling_rate)
                lim = min(70, len(psd_data[0]))
                self.psd_curves[i].setData(psd_data[1][:lim], psd_data[0][:lim])

                # Calculate band powers
                avg_bands[0] += DataFilter.get_band_power(psd_data, 1.0, 4.0)
                avg_bands[1] += DataFilter.get_band_power(psd_data, 4.0, 8.0)
                avg_bands[2] += DataFilter.get_band_power(psd_data, 8.0, 13.0)
                avg_bands[3] += DataFilter.get_band_power(psd_data, 13.0, 30.0)
                avg_bands[4] += DataFilter.get_band_power(psd_data, 30.0, 50.0)

        avg_bands = [x / len(self.exg_channels) for x in avg_bands]
        self.band_bar.setOpts(height=avg_bands)
        self.app.processEvents()


def main():
    # Enable logging
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = MindRoveInputParams()

    try:
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        Graph(board_shim)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == '__main__':
    main()