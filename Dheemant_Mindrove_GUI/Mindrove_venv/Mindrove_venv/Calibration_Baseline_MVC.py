
import sys
import os
import threading
import time
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QPushButton 
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QTimer
import pyqtgraph as pg
from Mindrove_get_data import MindRoveSensor
import pandas as pd
import numpy as np
from Filter_data import SignalProcessor
from datetime import datetime
import pandas as pd
from scipy.signal import find_peaks


class SensorThread(QThread):
    data_updated = pyqtSignal()

    def __init__(self, sensor):
        super().__init__()
        self.sensor = sensor
        self.running = False
        self.mutex = QMutex()

    def run(self):
        self.running = True
        self.sensor.start()
        while self.running:
            self.mutex.lock()
            self.data_updated.emit()
            self.mutex.unlock()
            time.sleep(0.01) 

    def stop(self):
        self.running = False
        self.sensor.stop()
        self.wait()



class ScrollingPlot(QMainWindow):
    def __init__(self, sensor, window_size=50, step_size=20):
        super().__init__()
        self.sensor = sensor
        self.window_size = window_size
        self.step_size = step_size

        # Initialize calibration-related variables
        self.mean_5s = [0] * self.sensor.num_channels
        self.std_5s = [0] * self.sensor.num_channels
        self.mvc_peak = [0] * self.sensor.num_channels

        # PyQtGraph setup for multiple channels
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.graph_widgets = []
        self.data_lines = []
        for channel in range(self.sensor.num_channels):
            graph_widget = pg.PlotWidget()
            graph_widget.setTitle(f"Channel {channel + 1}")
            graph_widget.setLabel("left", "Amp")
            graph_widget.setLabel("bottom", "Time")
            graph_widget.setLimits(xMin=0)
            
            # Make the plot widgets adaptable to resizing
            graph_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.graph_widgets.append(graph_widget)
            self.layout.addWidget(graph_widget)
            data_line = graph_widget.plot(pen=pg.mkPen(color='b', width=2))
            self.data_lines.append(data_line)

        # Start Button
        self.start_button = QPushButton("Start")
        self.layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.handle_start)
        
        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.layout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.handle_stop)

        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        
        # Start sensor data collection in a separate thread
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.data_updated.connect(self.update_plot)

    def handle_start(self):
        """Start the sensor and the timer."""
        print("Start button clicked. Starting sensor and timer...")
        # self.sensor.start()
        self.sensor_thread.start()
        print("Sensor thread started.")
        self.timer.start(20)  # Update every 20ms

    def segment_buffer(self):
        """Segment the buffer into overlapping windows, compute RMS, and prepare data for neural network training."""
        self.window_segments = {ch: [] for ch in range(self.sensor.num_channels)}
        all_windows = []

        processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        self.smoothed_buffers = processor.preproc_loop_buffer(self.window_size)

        for channel in range(self.sensor.num_channels):
            smoothed_buffer = self.smoothed_buffers[channel]

            if len(smoothed_buffer) >= self.window_size:
                for i in range(0, len(smoothed_buffer) - self.window_size + 1, self.step_size):
                    window = smoothed_buffer[i:i + self.window_size]
                    self.window_segments[channel].append((i, window))
                    all_windows.append({
                        "channel": channel,
                        "start_index": i,
                        "window": window
                    })
        return all_windows

    def update_plot(self):
        """Update the graph with new data and segmented windows."""
        all_windows = self.segment_buffer()
        
        for channel in range(self.sensor.num_channels):
            smoothed_buffer = self.smoothed_buffers[channel]
            
            if len(smoothed_buffer) > 0:
                for i, window in self.window_segments[channel]:
                    pass

            x_vals = range(len(smoothed_buffer))
            y_vals = smoothed_buffer
            self.data_lines[channel].setData(x_vals, y_vals)
            if len(x_vals) > self.window_size:
                self.graph_widgets[channel].setXRange(x_vals[-self.window_size], x_vals[-1])
            else:
                self.graph_widgets[channel].setXRange(0, self.window_size)

    def handle_stop(self):
        """Stop the timer, save data, and perform calibration."""
        print("Stop button clicked. Stopping timer and saving data...")
        
        # Stop the QTimer
        if self.timer.isActive():
            self.timer.stop()

        # Stop the sensor and any background threads
        self.sensor.stop()
        print("Sensor session stopped.")

        # Perform calibration using discarded data
        self.perform_calibration()

        # Exit the application
        print("Exiting application...")
        QApplication.quit()

    def perform_calibration(self):
        """Perform calibration using the discarded data."""
        discarded_data_df = self.sensor.get_discarded_elements()
        discarded_data_df = discarded_data_df.transpose()
        
        if not discarded_data_df.empty and len(discarded_data_df[0])==6000:
            print("Performing calibration using discarded data...")
            
            # Convert discarded data to numpy array
            discarded_data = discarded_data_df.to_numpy()
            
            # Preprocess the discarded data
            processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
            preprocessed_data = processor.preprocess_filters(discarded_data)
            
            # Calculate baseline mean and std
            for ch in range(self.sensor.num_channels):
                channel_data = preprocessed_data[ch][10:2500]  # Remove first 10 samples
                self.mean_5s[ch] = np.mean(channel_data)
                self.std_5s[ch] = np.std(channel_data)
            
            # Calculate MVC peak
            for ch in range(self.sensor.num_channels):
                ch_data = preprocessed_data[ch][2700:]
                peaks, _ = find_peaks(ch_data, height=0)
                if len(peaks) > 0:
                    self.mvc_peak[ch] = np.max(ch_data[peaks])
            
            print(f"Baseline Mean: {self.mean_5s}")
            print(f"Baseline Std: {self.std_5s}")
            print(f"MVC Peak Amplitudes: {self.mvc_peak}")
            
            # Save calibration data
            self.save_calibration()
        else:
            print("No discarded data available for calibration.")

    def save_calibration(self):
        """Save the calibration data to a CSV file."""
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"EMG_Calibration_{current_time}.csv"

        data_dict = {
            "Channel": [f"Ch{i+1}" for i in range(self.sensor.num_channels)],
            "Baseline Mean": self.mean_5s,
            "Baseline Std": self.std_5s,
            "MVC Peak": self.mvc_peak
        }

        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False)
        print(f"Calibration data saved to {filename}")
        


def main():
    # Initialize the sensor
    app = QApplication(sys.argv)
    print("QApplication initialized.")
    print("Starting main function...")
    sensor = MindRoveSensor()
    main_window = ScrollingPlot(sensor=sensor,window_size=200, step_size=100)

    
    print("Main window created.")
    main_window.show()
    print("Main window shown.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()