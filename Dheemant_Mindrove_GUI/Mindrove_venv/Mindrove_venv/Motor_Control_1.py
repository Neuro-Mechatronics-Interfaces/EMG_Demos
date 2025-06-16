# # ----------------------------------------------------------------------------------------


# # Works Fine (2/10/2025)

# # ----------------------------------------------------------------------------------------
# import os
# import sys  
# import time
# from datetime import datetime
# import threading
# import pandas as pd
# import numpy as np
# from collections import deque
# from PyQt5.QtGui import QPolygonF, QBrush, QColor
# from PyQt5.QtCore import Qt, QPointF, QThread, pyqtSignal, QMutex, QTimer
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QGridLayout,
#     QPushButton, QGraphicsPolygonItem, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox, QLabel, QLineEdit, QTextEdit
# )
# from scipy.signal import find_peaks
# import pyqtgraph as pg

# from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
# from Mindrove_get_data import MindRoveSensor
# from Filter_data import SignalProcessor
# from Single_thresh_Control import Single_thresh
# from Cursor_Control import GraphWidget
# from GUI_Plotter_RTMC_NML import ScrollingPlot 
# from Calibration_Baseline_MVC import ScrollingPlot as sp
# import re


# class SensorThread(QThread):
#     data_updated = pyqtSignal()

#     def __init__(self, sensor):
#         super().__init__()
#         self.sensor = sensor
#         self.running = False
#         self.mutex = QMutex()

#     def run(self):
#         self.running = True
#         self.sensor.start()
#         while self.running:
#             self.mutex.lock()
#             self.data_updated.emit()
#             self.mutex.unlock()
#             time.sleep(0.01) 

#     def stop(self):
#         self.running = False
#         self.sensor.stop()
#         self.wait()
# class MyApp(QMainWindow):
#     def __init__(self, sensor, window, window_size=200, step_size=100):
#         super().__init__()
        
#         self.sensor = sensor
#         self.window_size = window_size
#         self.step_size = step_size
#         self.window = window
        
#         self.setWindowTitle("NML-MindRove RTMC-GUI")
#         self.setGeometry(200, 200, 2000, 1500)

#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.main_layout = QHBoxLayout(self.central_widget)

#         self.left_layout = QVBoxLayout()

#         today_str = datetime.now().strftime('%Y_%m_%d')
#         default_filename = f"data_{today_str}"  
        
#         self.left_layout.addWidget(QLabel("Filename:"))
#         self.filename_input = QLineEdit(default_filename)
#         self.left_layout.addWidget(self.filename_input)
        
#         self.left_layout.addWidget(QLabel("Baseline:"))
#         self.Baseline_input = QLineEdit("[0,0,0,0,0,0,0,0]")
#         self.left_layout.addWidget(self.Baseline_input)
        
#         self.left_layout.addWidget(QLabel("MVC:"))
#         self.mvc_input = QLineEdit("[0,0,0,0,0,0,0,0]")
#         self.left_layout.addWidget(self.mvc_input)
        
#         self.left_layout.addWidget(QLabel("Percentage:"))
#         self.percentage_input = QLineEdit("20")
#         self.left_layout.addWidget(self.percentage_input)
        
#         self.filename_input.textChanged.connect(self.handle_file_status_update)
#         self.Baseline_input.textChanged.connect(self.on_calibrate_click)
#         self.mvc_input.textChanged.connect(self.on_calibrate_mvc_click)
        
#         self.save_checkbox = QCheckBox("Save to File")
#         self.save_checkbox.setChecked(True)
#         self.left_layout.addWidget(self.save_checkbox)
#         self.save_checkbox.clicked.connect(self.handle_file_status_update)
        
#         self.left_layout.addWidget(QLabel("File Suffix:"))
#         self.suffix_spinbox = QSpinBox()
#         self.suffix_spinbox.setMinimum(0)
#         self.suffix_spinbox.setValue(0)
#         self.left_layout.addWidget(self.suffix_spinbox)
#         self.suffix_spinbox.valueChanged.connect(self.handle_file_status_update)
        
#         self.calibrate_baseline_button = QPushButton('Calibrate Baseline', self)
#         self.calibrate_mvc_button = QPushButton('Calibrate MVC', self)
#         self.start_button = QPushButton('Start', self)
#         self.stop_button = QPushButton('Stop', self)
#         self.save_button = QPushButton('Save', self)

#         self.calibrate_baseline_button.setStyleSheet("background-color: yellow;")
#         self.calibrate_mvc_button.setStyleSheet("background-color: orange;")
#         self.start_button.setStyleSheet("background-color: green;")
#         self.stop_button.setStyleSheet("background-color: red;")
#         self.save_button.setStyleSheet("background-color: yellow;")

#         button_layout = QHBoxLayout()
#         button_layout.addWidget(self.calibrate_baseline_button)
#         button_layout.addWidget(self.calibrate_mvc_button)
#         button_layout.addWidget(self.start_button)
#         button_layout.addWidget(self.stop_button)
#         button_layout.addWidget(self.save_button)
#         button_layout.setAlignment(Qt.AlignCenter)

#         self.left_layout.addLayout(button_layout)
        
#         self.stimulus_display = QTextEdit()
#         self.stimulus_display.setReadOnly(True)
#         self.stimulus_display.setAlignment(Qt.AlignCenter)
#         self.update_stimulus("Stay still for baseline!", "white", "black")
#         self.left_layout.addWidget(self.stimulus_display)
        
#         self.left_layout.addStretch()
#         self.main_layout.addLayout(self.left_layout)

#         self.right_layout = QVBoxLayout()

#         self.graph_widgets = []
#         self.data_lines = []
        
        
        
#         for channel in range(8):
#             graph_widget = pg.PlotWidget()
#             graph_widget.setTitle(f"Channel {channel + 1}")
#             graph_widget.setLabel("left", "Amp")
#             graph_widget.setLabel("bottom", "Time")
#             graph_widget.setLimits(xMin=0)

#             graph_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#             self.graph_widgets.append(graph_widget)
#             self.right_layout.addWidget(graph_widget)
#             data_line = graph_widget.plot(pen=pg.mkPen(color='b', width=2))
#             self.data_lines.append(data_line)

#         self.main_layout.addLayout(self.right_layout)

#         self.calibrate_baseline_button.clicked.connect(self.on_calibrate_click)
#         self.calibrate_mvc_button.clicked.connect(self.on_calibrate_mvc_click)
#         self.start_button.clicked.connect(self.on_start_click)
#         self.stop_button.clicked.connect(self.on_stop_click)
#         self.save_button.clicked.connect(self.on_save_click)
        
#         self.mean_5s = np.zeros(8)
#         self.std_5s = np.zeros(8)
#         self.mvc_peak = np.zeros(8)
        
#         # Timer for real-time updates
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_plot)
        
#         # Sensor thread for data collection
#         self.sensor_thread = SensorThread(self.sensor)
        
#         # MVC calibration state
#         self.mvc_calibrating = False
#         self.mvc_data = [[] for _ in range(8)]  # Buffer to store MVC data
#         self.threshold_calculator = Single_thresh(self.sensor, self.window_size, self.step_size, self.mvc_peak)
        
#     def update_stimulus(self, text, bg_color, text_color):
#         self.stimulus_display.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; font-size: 50px;")
#         self.stimulus_display.setText(text)
        
#     def segment_buffer(self):
#         """Segment the buffer into overlapping windows, apply Min-Max normalization, and prepare data for neural network training."""
#         self.window_segments = {ch: [] for ch in range(self.sensor.num_channels)}
#         all_windows = []

#         processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
#         self.smoothed_buffers = processor.preproc_loop_buffer(self.window_size)
        
#         # Extract Baseline and MVC values as lists
#         baseline_values = list(map(float, self.Baseline_input.text().strip("[]").split()))  
#         mvc_values = list(map(float, self.mvc_input.text().strip("[]").split()))

#         for channel in range(self.sensor.num_channels):
#             smoothed_buffer = np.array(self.smoothed_buffers[channel])

#             # Apply Min-Max Normalization: (X - baseline) / (MVC - baseline)
#             if mvc_values[channel] - baseline_values[channel] != 0:
#                 normalized_buffer = (smoothed_buffer - baseline_values[channel]) / (mvc_values[channel] - baseline_values[channel])
#             else:
#                 normalized_buffer = smoothed_buffer  # Avoid division by zero error
                
#             # print(normalized_buffer)

#             self.smoothed_buffers[channel] = normalized_buffer  # Store normalized data

#             if len(normalized_buffer) >= self.window_size:
#                 for i in range(0, len(normalized_buffer) - self.window_size + 1, self.step_size):
#                     window = normalized_buffer[i:i + self.window_size]
#                     self.window_segments[channel].append((i, window))
#                     all_windows.append({
#                         "channel": channel,
#                         "start_index": i,
#                         "window": window
#                     })

#         return all_windows


#     def update_plot(self):
#         """Update the graph with new data and segmented windows."""
#         mvc_text = self.mvc_input.text().strip("[]")  # Remove brackets
#         mvc_values = list(map(float, mvc_text.split()))  # Split on spaces
        
#         all_windows = self.segment_buffer()
        
#         for channel in range(self.sensor.num_channels):
#             smoothed_buffer = self.smoothed_buffers[channel]
            
#             if len(smoothed_buffer) > 0:
#                 for i, window in self.window_segments[channel]:
#                     movement_status = self.threshold_calculator.Thresh_Implement(window, int(self.percentage_input.text()), mvc_values)
#                     # print(f"Movement status: {movement_status}")
#                     self.window.set_movement_status(movement_status)
#                     self.window.update()
#                     pass

#             x_vals = range(len(smoothed_buffer))
#             y_vals = smoothed_buffer
#             self.data_lines[channel].setData(x_vals, y_vals)
#             if len(x_vals) > self.window_size:
#                 self.graph_widgets[channel].setXRange(x_vals[-self.window_size], x_vals[-1])
#             else:
#                 self.graph_widgets[channel].setXRange(0, self.window_size)

#     def on_calibrate_click(self):
#         """Start baseline calibration when the button is clicked."""
#         self.update_stimulus("Stay still for baseline!", "white", "black")
#         print("Starting baseline calibration...")
#         self.sensor.do_baseline = True

#         # Start the sensor thread and timer
#         self.sensor_thread = SensorThread(self.sensor)
#         self.sensor_thread.start()
#         self.timer.start(20)

#         # Wait for sufficient data
#         while not all(len(self.sensor.baseline[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
#             print("Waiting for sufficient baseline data...")
#             time.sleep(0.1) 
#         # start_time = time.time()
#         # while time.time() - start_time < 6:
#         #     # discarded_data_df = self.sensor.get_discarded_elements().transpose()
#         #     if all(len(self.sensor.baseline[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
#         #         break
            
#         #     print("Waiting for sufficient discarded data...")
#         #     time.sleep(0.1)  # Prevents busy-waiting
            
#         # print(self.sensor.baseline[0])
#         # Check if we got enough data
#         # if any(len(self.sensor.baseline[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
#         #     print("Error: Insufficient discarded data for calibration.")
#         #     return
        
#         # for ch in range(self.sensor.num_channels):
#         #     print(f"Channel {ch} data type: {type(self.sensor.baseline[ch])}")
#         #     print(f"Channel {ch} data length: {len(self.sensor.baseline[ch])}")
            
#         print("Performing calibration using discarded data...")
#         baseline_data = np.array([list(self.sensor.baseline[ch])[:2500] for ch in range(self.sensor.num_channels)])
#         processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
#         preprocessed_data = processor.preprocess_filters(baseline_data)
        
#         for ch in range(self.sensor.num_channels):
#             channel_data = preprocessed_data[ch][0:2500]
#             self.mean_5s[ch] = np.mean(channel_data)
#             self.std_5s[ch] = np.std(channel_data)
            
#         self.Baseline_input.setText(f"{self.mean_5s}")
#         print(f"Baseline Mean: {self.mean_5s}")
#         print(f"Baseline Std: {self.std_5s}")
        
#         # Stop the sensor thread after calibration
#         self.sensor_thread.stop()
#         self.timer.stop()
        
#         self.sensor.do_baseline = False
        
#         self.update_stimulus("Baseline Calibrated! Now calibrate MVC.", "yellow", "black")
        
        
#         for ch in range(self.sensor.num_channels):
#             self.sensor.baseline[ch] = []


#     def on_calibrate_mvc_click(self):
#         """Start MVC calibration when the button is clicked."""
#         self.update_stimulus("Performing MVC calibration...", "orange", "black")
#         print("Starting MVC calibration...")
#         self.sensor.do_mvc = True

#         # Start the sensor thread and timer
#         self.sensor_thread = SensorThread(self.sensor)
#         self.sensor_thread.start()
#         self.timer.start(20)

#         # Wait for sufficient data
#         while not all(len(self.sensor.mvc[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
#             print("Waiting for sufficient MVC data...")
#             time.sleep(0.1)
#         # start_time = time.time()
#         # while time.time() - start_time < 6:  # Ensure enough time for MVC collection
#         #     if all(len(self.sensor.mvc[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
#         #         break
#         #     print("Waiting for sufficient MVC data...")
#         #     time.sleep(0.1)

#         # # Ensure sufficient data is collected
#         # if any(len(self.sensor.mvc[ch]) < 2500 for ch in range(self.sensor.num_channels)):
#         #     print("Error: Insufficient MVC data for calibration.")
#         #     return

#         print("Processing MVC data...")
#         mvc_data = np.array([list(self.sensor.mvc[ch])[:2500] for ch in range(self.sensor.num_channels)])

#         # Apply signal processing
#         processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
#         preprocessed_data = processor.preprocess_filters(mvc_data)

#         # Compute MVC peak amplitudes
#         for ch in range(self.sensor.num_channels):
#             channel_data = preprocessed_data[ch][0:2500]  # Using data after 2510 for peak calculation
#             if len(channel_data) > 0:
#                 peaks, _ = find_peaks(channel_data, height=0)
#                 self.mvc_peak[ch] = np.max(channel_data[peaks]) if len(peaks) > 0 else 0

#         self.mvc_input.setText(f"{self.mvc_peak}")
#         print(f"MVC Peak Amplitudes: {self.mvc_peak}")

#         # Stop the sensor thread and clear data
#         self.sensor_thread.stop()
#         self.timer.stop()
#         self.sensor.do_mvc = False

#         # Clear MVC data after calibration
#         for ch in range(self.sensor.num_channels):
#             self.sensor.mvc[ch] = []

#         self.update_stimulus("MVC Calibration Complete!", "green", "black")


#     def on_start_click(self):
#         """Start the sensor stream for real-time monitoring."""
#         self.sensor.collect_data = True
#         self.update_stimulus("Go!", "green", "black")
#         self.sensor_thread = SensorThread(self.sensor)
#         self.sensor_thread.start()
#         self.timer.start(20)

#     def on_stop_click(self):
#         self.sensor.collect_data = False
#         """Stop the sensor stream."""
#         self.update_stimulus("Stay still for baseline!", "white", "black")
#         self.sensor_thread.stop()
#         self.timer.stop()

#     def on_save_click(self):
#         """Save the data to a file."""
#         print("Save button clicked!")
        
#         if self.save_checkbox.isChecked():
#             print("Discarded Elements DataFrame:")
#             discarded_data_df = self.sensor.get_discarded_elements()
#             if not discarded_data_df.empty:
#                 filename = self.filename_input.text()
#                 suffix = self.suffix_spinbox.value()
#                 full_filwswswswswswswswswswename = f"{filename}_{suffix}.csv"
#                 filepath = os.path.join(os.getcwd(), full_filename)
#                 print(f"Saving data to: {filepath}")
#                 try:
#                     discarded_data_df.to_csv(filepath, index=False)
#                     print(f"Discarded data successfully saved to {filepath}.")
#                 except Exception as e:
#                     print(f"An error occurred while saving the discarded data: {e}")
#         else:
#             print("No discarded data to save.")

#     def handle_file_status_update(self):
#         """Handle updates to the filename, suffix, or save status."""
#         filename = self.filename_input.text()
#         suffix = self.suffix_spinbox.value()
#         save_status = self.save_checkbox.isChecked()
#         print(f"Filename: {filename}, Suffix: {suffix}, Save Status: {save_status}")
        




# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     sensor = MindRoveSensor(buffer_size=300, num_channels=8)
#     Cursor =GraphWidget()
#     window = MyApp(sensor=sensor, window = Cursor)
#     window.show()
#     Cursor.show()
#     sys.exit(app.exec_())


# import serial
# import keyboard
# import time

# arduino = serial.Serial('COM9', 9600)  # Adjust COM port
# time.sleep(2)  # Allow time for connection

# while True:
#     if keyboard.is_pressed('w'):
#         arduino.write(b'W')
#         print("Sent: W")
#         time.sleep(0.2)  # Prevent repeated sends
#     elif keyboard.is_pressed('s'):
#         arduino.write(b'S')
#         print("Sent: S")
#         time.sleep(0.2)  # Prevent repeated sends


import time
import serial

arduino = serial.Serial('COM9', 9600)
for angle in [0, 45, 90, 135, 180]:
    print(f"Sending angle: {angle}")
    arduino.write(str(angle).encode())
    arduino.flush()
    time.sleep(1.5)
arduino.close()