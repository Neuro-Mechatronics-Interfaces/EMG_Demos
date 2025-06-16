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
from Single_thresh_Control import Single_thresh
from Cursor_Control import GraphWidget


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
    def __init__(self, sensor, window,window_size=50, step_size=20):
        super().__init__()
        self.sensor = sensor
        self.window_size = window_size
        self.step_size = step_size
        self.window = window

        self.threshold_calculator = Single_thresh(self.sensor, self.window_size, self.step_size)
        
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
        # self.timer.start(20)  # Update every 10ms
        
        # Start sensor data collection in a separate thread
        # threading.Thread(target=self.sensor.start, daemon=True).start() # Old Alternative
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.data_updated.connect(self.update_plot)
        # self.sensor_thread.start()

    
    def handle_start(self):
        """Start the sensor and the timer."""
        print("Start button clicked. Starting sensor and timer...")
        self.sensor_thread.start()
        self.timer.start(20)  # Update every 20ms
        
    def segment_buffer(self):
        """Segment the buffer into overlapping windows, compute RMS, and prepare data for neural network training."""
        # Initialize dictionaries to store overlapping segments and metadata
        self.window_segments = {ch: [] for ch in range(self.sensor.num_channels)}  # For overlapping segmentation
        all_windows = []  # List to store all window data for all channels

        # Initialize the signal processor
        processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        self.smoothed_buffers = processor.preproc_loop_buffer(self.window_size)
        
        baseline = [217.05008562080698,182.69425626386212,171.49950737470982, 308.02264557881017,182.7220181579753, 192.3978575914023,128.6835717611569, 197.81808042678566]
        
        
        
        
        

        # Loop through each channel to perform overlapping segmentation
        for channel in range(self.sensor.num_channels):
            smoothed_buffer =np.array(self.smoothed_buffers[channel])-baseline[channel]

            # Perform overlapping segmentation
            if len(smoothed_buffer) >= self.window_size:
                for i in range(0, len(smoothed_buffer) - self.window_size + 1, self.step_size):
                    window = smoothed_buffer[i:i + self.window_size]
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    self.window_segments[channel].append((i, window))  # Store index and window
                    
                    # Store the window data in a list for Pandas DataFrame creation
                    all_windows.append({
                        "channel": channel,
                        "start_index": i,
                        "window": window
                    })
        return all_windows

    def Save_data(self, data):
        if not data:
            print("No data to save.")
            return
        try:
            df = pd.DataFrame(data)
            if df.empty:
                print("DataFrame is empty.")
                return

            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            base_filename = "Real_time_testing"
            filename = f"{base_filename}_{formatted_time}.csv"
            
            print(f"Attempting to save data to {filename}")
            df.to_csv(filename, index=False)
            print(f"Data successfully saved to {filename}")
        except Exception as e:
            print(f"An error occurred while saving the data: {e}")
        
    def update_plot(self):
        """Update the graph with new data and segmented windows."""
        all_windows = self.segment_buffer()
        
        # Update plots for each channel
        for channel in range(self.sensor.num_channels):
            # preprocessed_buffer = self.preprocessed_buffers[channel]
            smoothed_buffer = self.smoothed_buffers[channel]
            # buffer = list(self.sensor.get_buffer(channel))
            
            if len(smoothed_buffer) > 0:
            # if len(preprocessed_buffer) > 0:
            # if buffer:
                # print(f"Channel {channel + 1} Buffer: {buffer}")  # Print buffer contents
                for i, window in self.window_segments[channel]:
                    # print(f"Channel {channel + 1} Window {i}-{i+self.window_size}: {window}")
                    
                    movement_status = self.threshold_calculator.Thresh_Implement(window, 1)
                    print(f"Movement status: {movement_status}")
                    self.window.set_movement_status(movement_status)
                    self.window.update()
                    pass
                    

            # Update the plot
            x_vals = range(len(smoothed_buffer))
            y_vals = smoothed_buffer
            self.data_lines[channel].setData(x_vals, y_vals)
            if len(x_vals) > self.window_size:
                self.graph_widgets[channel].setXRange(x_vals[-self.window_size], x_vals[-1])
            else:
                self.graph_widgets[channel].setXRange(0, self.window_size)

    def handle_stop(self):
        """Stop the timer, save data, and exit the application."""
        print("Stop button clicked. Stopping timer and saving data...")
        
        # Save discarded data
        discarded_data_df = self.sensor.get_discarded_elements()
        print("Discarded Elements DataFrame:")
        print(discarded_data_df) 

        if not discarded_data_df.empty:
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"discarded_data_{formatted_time}.csv"
            filepath = os.path.join(os.getcwd(), filename)
            print(f"Attempting to save discarded data to: {filepath}")

            try:
                discarded_data_df.to_csv(filepath, index=False)
                print(f"Discarded data successfully saved to {filepath}.")
            except Exception as e:
                print(f"An error occurred while saving the discarded data: {e}")
        else:
            print("No discarded data to save.")
        
        # Stop the QTimer
        if self.timer.isActive():
            self.timer.stop()

        # Stop the sensor and any background threads
        self.sensor.stop()
        print("Sensor session stopped.")

        # Exit the application
        print("Exiting application...")
        QApplication.quit()        

def main():
    app = QApplication(sys.argv)

    # Initialize the sensor with 8 channels
    sensor = MindRoveSensor(buffer_size=1000, num_channels=8)
    window = GraphWidget()
    movement_timer = QTimer()
    movement_timer.timeout.connect(Single_thresh.Thresh_Implement)
    
    

    # Start the scrolling plot
    plot_window = ScrollingPlot(sensor=sensor, window = window ,window_size=200, step_size=100)
    print("Starting the application. Use the 'Stop' button to stop and save discarded data.")
    
    plot_window.show()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()

