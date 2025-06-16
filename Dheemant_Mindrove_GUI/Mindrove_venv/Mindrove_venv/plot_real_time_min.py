# ------------------------------------------------------------------------------------------


# MINDROVE (PLOT_REAL_TIME_MIN.PY)_____________________________WORKING


# ------------------------------------------------------------------------------------------
import argparse
import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title='Mindrove Plot')
        self.win.resize(800, 600)

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)

        for count, channel in enumerate(self.exg_channels):
            # Plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                data[channel],
                self.sampling_rate,
                51.0,
                100.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,)
            # )
            # DataFilter.perform_bandstop(
            #     data[channel],
            #     self.sampling_rate,
            #     50.0,
            #     4.0,
            #     2,
            #     FilterTypes.BUTTERWORTH.value,
            #     0,
            # )
            # DataFilter.perform_bandstop(
            #     data[channel],
            #     self.sampling_rate,
            #     60.0,
            #     4.0,
            #     2,
            #     FilterTypes.BUTTERWORTH.value,
            #     0,
            # )
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = MindRoveInputParams()

    try:
        board_shim = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
        board_shim.prepare_session()
        board_shim.start_stream()
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()






