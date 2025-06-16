#------------------------------------------------

# Freeze project .. waiting for Communication protocol to be integrated
# 3/12/25

#------------------------------------------------
import os
import sys  
import time
from datetime import datetime
import threading
import pandas as pd
import numpy as np
from collections import deque
from PyQt5.QtGui import QPolygonF, QBrush, QColor
from PyQt5.QtCore import Qt, QPointF, QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QGridLayout,
    QPushButton, QGraphicsPolygonItem, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox, QLabel, QLineEdit, QTextEdit, QComboBox, QInputDialog
)
from scipy.signal import find_peaks
import pyqtgraph as pg

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from Mindrove_get_data import MindRoveSensor
from Filter_data import SignalProcessor
from Single_thresh_Control import Controls_st_pc
from Cursor_Control import GraphWidget
from Calibration_Baseline_MVC import ScrollingPlot as sp
import re
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt

class StimulusThread(QThread):
    # Define signals to communicate with the main thread
    update_stim_signal = pyqtSignal(str, str, str)
    
    def __init__(self, count=10, rest_duration=0.5, up_duration=0.5, hold_duration=0.5, down_duration=0.5):
        super().__init__()
        self.running = True
        self.count = count
        self.rest_duration = rest_duration
        self.up_duration = up_duration
        self.hold_duration = hold_duration
        self.down_duration = down_duration
    
    def run(self):
        """Main execution method for the thread"""
        start_time = time.time()
        count = 0
        
        # Start with "Rest"
        self.update_stim_signal.emit('Stay Still', 'white', 'black')
        last_change_time = start_time
        current_state = "rest"
        
        while count < self.count and self.running:
            current_time = time.time()
            elapsed_since_change = current_time - last_change_time
            
            # Get the duration for the current state
            if current_state == "rest":
                current_duration = self.rest_duration
            elif current_state == "up":
                current_duration = self.up_duration
            elif current_state == "hold":
                current_duration = self.hold_duration
            elif current_state == "down":
                current_duration = self.down_duration
            
            # Check if the state's duration has passed
            if elapsed_since_change >= current_duration:
                # Change to the next state in the cycle
                if current_state == "rest":
                    self.update_stim_signal.emit('Ramp Up', 'white', 'black')
                    current_state = "up"
                elif current_state == "up":
                    self.update_stim_signal.emit('Hold', 'white', 'black')
                    current_state = "hold"
                elif current_state == "hold":
                    self.update_stim_signal.emit('Ramp Down', 'white', 'black')
                    current_state = "down"
                elif current_state == "down":
                    self.update_stim_signal.emit('Stay Still', 'white', 'black')
                    current_state = "rest"
                    count += 1  # Increment count after completing a full cycle
                
                # Update the time of last change
                last_change_time = current_time
                print(f"Stimulus changed to: {current_state}, count: {count}")
            
            # Use QThread's msleep instead of time.sleep
            self.msleep(10)  # 10ms sleep
        
        self.update_stim_signal.emit('Complete', 'green', 'white')

    def stop(self):
        """Stop the thread."""
        self.running = False
        
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

class MyApp(QMainWindow):
    def __init__(self, sensor, window, window_size=200, step_size=100):
        super().__init__()
        
        self.setStyleSheet("background-color: black; color: white;")
        self.sensor = sensor
        self.window_size = window_size
        self.step_size = step_size
        self.window = window
        self.stimulus_thread = None
        self.setWindowTitle("NML-MindRove RTMC-GUI")
        self.setGeometry(200, 200, 3000, 1500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create and configure a QLabel to hold the background image
        self.bg_image_label = QLabel(self.central_widget)
        self.bg_image_label.setAlignment(Qt.AlignLeft| Qt.AlignBottom)

        image = QPixmap("D:/CMU_NML/Mindrove_venv/NML_logo.png")

        transparent_image = QPixmap(image.size())
        transparent_image.fill(Qt.transparent)
        painter = QPainter(transparent_image)
        painter.setOpacity(0.5)  
        painter.drawPixmap(0, 0, image)
        painter.end()
        
        # Set the image to the label
        self.bg_image_label.setPixmap(transparent_image)
        self.bg_image_label.setGeometry(600, 1350, image.width(), image.height())
        
        
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()

        today_str = datetime.now().strftime('%Y_%m_%d')
        default_filename = f"data_{today_str}"  
        
        self.left_layout.addWidget(QLabel("Filename:"))
        self.filename_input = QLineEdit(default_filename)
        self.left_layout.addWidget(self.filename_input)
        
        self.left_layout.addWidget(QLabel("Baseline:"))
        self.Baseline_input = QLineEdit("[0 0 0 0 0 0 0 0]")
        self.left_layout.addWidget(self.Baseline_input)

        self.mvc_display_label = QLabel("Action MVC Values:")
        self.left_layout.addWidget(self.mvc_display_label)
        self.mvc_display = QTextEdit()
        self.mvc_display.setReadOnly(True)
        self.mvc_display.setMaximumHeight(80)
        self.left_layout.addWidget(self.mvc_display)
        
        # Create a horizontal layout for the checkboxes
        checkbox_layout = QHBoxLayout()
        
        # Add checkboxes for selecting control implementation
        self.sing_thres_checkbox = QCheckBox("Single Threshold Control")
        self.pro_cont_checkbox = QCheckBox("Proportional Control")

        # Ensure only one can be selected at a time
        self.sing_thres_checkbox.toggled.connect(self.update_control_selection)
        self.pro_cont_checkbox.toggled.connect(self.update_control_selection)

        checkbox_layout.addWidget(self.sing_thres_checkbox)
        checkbox_layout.addWidget(self.pro_cont_checkbox)
        
        # Dictionary to store RMS values for each dropdown option
        self.option_rms_values = {}
        # Dictionary to store the selected channel index for each option
        self.option_selected_channels = {}
        self.option_mvc_values = {}
        
        self.control_dropdown = QComboBox()
        # Initialize with default options and their corresponding blank lists
        default_options = ["Wflex", "WExten"]
        for option in default_options:
            self.control_dropdown.addItem(option)
            self.option_rms_values[option] = []
            self.option_selected_channels[option] = 0
            self.option_mvc_values[option] = [0] * 8  
        
        # Button to add new options dynamically
        self.add_option_button = QPushButton("+")
        self.add_option_button.setFixedWidth(30)
        self.add_option_button.clicked.connect(self.add_new_option)
        
        # Connect dropdown selection change event
        self.control_dropdown.currentTextChanged.connect(self.on_dropdown_changed)
        
        # Motor_Element_Actuation:
        self.motor_dropdown = QComboBox()
        def_op_motor = ["movemotor:1:{};", "movemotor:2:{};","set_gripper:{};"]
        for option in def_op_motor:
            self.motor_dropdown.addItem(option)
        
        # Button to add new options dynamically
        self.add_option_button_motor = QPushButton("+motor")
        self.add_option_button_motor.setFixedWidth(100)
        self.add_option_button_motor.clicked.connect(self.add_new_option_motor)

        checkbox_layout.addWidget(self.control_dropdown)
        checkbox_layout.addWidget(self.add_option_button)
        checkbox_layout.addWidget(self.motor_dropdown)
        checkbox_layout.addWidget(self.add_option_button_motor)
        
        # Add a text display for the current selection's RMS values
        self.rms_display_label = QLabel("RMS Values:")
        self.left_layout.addWidget(self.rms_display_label)
        self.rms_display = QTextEdit()
        self.rms_display.setReadOnly(True)
        self.rms_display.setMaximumHeight(80)
        self.left_layout.addWidget(self.rms_display)
        
        # Add a label to show currently selected channel for the movement
        self.selected_channel_label = QLabel("Selected Channel: None")
        self.left_layout.addWidget(self.selected_channel_label)
        
        # Default selections
        self.sing_thres_checkbox.setChecked(False)  # Default to single threshold
        self.pro_cont_checkbox.setChecked(True)
        self.current_control_method = "proportional_control"   # Default method
        self.left_layout.addLayout(checkbox_layout)
        
        Param_layout = QHBoxLayout()
        Param_layout.addWidget(QLabel("Percentage:"))
        self.percentage_input = QLineEdit("20")
        Param_layout.addWidget(self.percentage_input)
        
        Param_layout.addWidget(QLabel("Angle Tolerance:"))
        self.update_diffangle = QLineEdit("2.5")
        Param_layout.addWidget(self.update_diffangle)
        
        Param_layout.addWidget(QLabel("Time Tolerance:"))
        self.update_difftime = QLineEdit("1.5")
        Param_layout.addWidget(self.update_difftime)
        
        self.percentage_input.textChanged.connect(self.update_parameters)
        self.update_diffangle.textChanged.connect(self.update_parameters)
        self.update_difftime.textChanged.connect(self.update_parameters)
        
        self.left_layout.addLayout(Param_layout)
        
        self.filename_input.textChanged.connect(self.handle_file_status_update)
        self.Baseline_input.textChanged.connect(self.on_calibrate_click)
        # self.mvc_input.textChanged.connect(self.on_calibrate_mvc_click)
        
        self.save_layout = QHBoxLayout()
        
        self.save_checkbox = QCheckBox("Save to File")
        self.save_checkbox.setChecked(True)
        self.save_layout.addWidget(self.save_checkbox)
        self.save_checkbox.clicked.connect(self.handle_file_status_update)
        self.left_layout.addLayout(self.save_layout)
        
        self.left_layout.addWidget(QLabel("File Suffix:"))
        self.suffix_spinbox = QSpinBox()
        self.suffix_spinbox.setMinimum(0)
        self.suffix_spinbox.setValue(0)
        self.left_layout.addWidget(self.suffix_spinbox)
        self.suffix_spinbox.valueChanged.connect(self.handle_file_status_update)
        
        self.calibrate_baseline_button = QPushButton('Calibrate Baseline', self)
        self.calibrate_action_mvc_button = QPushButton('Calibrate Action MVC', self)
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.save_button = QPushButton('Save', self)
        self.smart_select = QPushButton('Smart Select', self)
        self.smart_select_save = QPushButton('Smart Select Save', self)

        self.calibrate_baseline_button.setStyleSheet("background-color: yellow; color: black;")
        self.calibrate_action_mvc_button.setStyleSheet("background-color: orange; color: black;")
        self.start_button.setStyleSheet("background-color: green; color: black;")
        self.stop_button.setStyleSheet("background-color: red; color: black;")
        self.save_button.setStyleSheet("background-color: yellow; color: black;")
        self.smart_select.setStyleSheet("background-color: yellow; color: black;")
        self.smart_select_save.setStyleSheet("background-color: yellow; color: black;")
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.calibrate_baseline_button)
        button_layout.addWidget(self.calibrate_action_mvc_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.smart_select)
        button_layout.addWidget(self.smart_select_save)
        button_layout.setAlignment(Qt.AlignCenter)

        self.left_layout.addLayout(button_layout)

        
        # Mean MVC peak for the currently selected action
        self.action_mvc_peak = np.zeros(8)
        
        # Stimulus Display
        self.stimulus_display = QTextEdit()
        self.stimulus_display.setReadOnly(True)
        self.stimulus_display.setAlignment(Qt.AlignCenter)
        self.update_stimulus("Stay still for baseline!", "white", "black")
        self.left_layout.addWidget(self.stimulus_display)

        # Create a layout for count and duration inputs below the stimulus display
        self.calibration_timer = None
        self.count_input = QLineEdit("10")  # Default value 10
        self.rest_duration_input = QLineEdit("0.5")  # Default value 0.5 seconds
        self.up_duration_input = QLineEdit("0.5")
        self.hold_duration_input = QLineEdit("0.5")
        self.down_duration_input = QLineEdit("0.5")
        
        self.duration_layout = QHBoxLayout()

        def create_label_input_pair(label_text, input_widget):
            layout = QVBoxLayout()
            layout.addWidget(QLabel(label_text))
            layout.addWidget(input_widget)
            return layout

        self.duration_layout.addLayout(create_label_input_pair("Count:", self.count_input))
        self.duration_layout.addLayout(create_label_input_pair("Rest Duration (s):", self.rest_duration_input))
        self.duration_layout.addLayout(create_label_input_pair("Up Duration (s):", self.up_duration_input))
        self.duration_layout.addLayout(create_label_input_pair("Hold Duration (s):", self.hold_duration_input))
        self.duration_layout.addLayout(create_label_input_pair("Down Duration (s):", self.down_duration_input))
        self.left_layout.addLayout(self.duration_layout)
        
        
        self.force_select_layout = QVBoxLayout()

        # Add the force selection layout to the left layout (at the bottom)
        self.force_select_layout.addWidget(QLabel("Force Choose Channel:"))
        self.force_channel_dropdown = QComboBox()
        force_channel_options = ["None", "1", "2", "3", "4", "5", "6", "7", "8"]
        self.force_channel_dropdown.addItems(force_channel_options)
        self.force_select_layout.addWidget(self.force_channel_dropdown)
        self.left_layout.addLayout(self.force_select_layout)

        self.left_layout.addStretch()
        self.main_layout.addLayout(self.left_layout)

        self.right_layout = QVBoxLayout()

        self.graph_widgets = []
        self.data_lines = []
        
        for channel in range(8):
            graph_widget = pg.PlotWidget()
            graph_widget.setTitle(f"Channel {channel + 1}")
            graph_widget.setLabel("left", "Amp")
            graph_widget.setLabel("bottom", "Time")
            graph_widget.setLimits(xMin=0)
            
            graph_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.graph_widgets.append(graph_widget)
            self.right_layout.addWidget(graph_widget)
            data_line = graph_widget.plot(pen=pg.mkPen(color='b', width=2))
            self.data_lines.append(data_line)

        self.main_layout.addLayout(self.right_layout)

        self.calibrate_baseline_button.clicked.connect(self.on_calibrate_click)
        self.calibrate_action_mvc_button.clicked.connect(self.on_calibrate_action_mvc_click)
        self.start_button.clicked.connect(self.on_start_click)
        self.stop_button.clicked.connect(self.on_stop_click)
        self.save_button.clicked.connect(self.on_save_click)
        self.smart_select.clicked.connect(self.smart_select_bf)
        self.smart_select_save.clicked.connect(self.smart_select_save_bf)
        
        self.mean_5s = np.zeros(8)
        self.std_5s = np.zeros(8)
        self.mvc_peak = np.zeros(8)
        
        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        
        # Sensor thread for data collection
        self.sensor_thread = SensorThread(self.sensor)
        
        # MVC calibration state
        self.mvc_calibrating = False
        self.mvc_data = [[] for _ in range(8)]  # Buffer to store MVC data
        try:
            self.current_percentage = int(self.percentage_input.text())
            self.current_angle_tolerance = float(self.update_diffangle.text())
            self.current_time_tolerance = float(self.update_difftime.text())
        except ValueError:
            self.current_percentage = 20
            self.current_angle_tolerance = 2.5
            self.current_time_tolerance = 1.5
        self.ind_chan = 1
        self.threshold_calculator = Controls_st_pc(self.sensor, self.window_size, self.step_size, self.mvc_peak, 
                                                 self.ind_chan, self.current_percentage, 
                                                 self.current_angle_tolerance, self.current_time_tolerance, self)
        
        
    def update_control_selection(self):
        """Ensure only one control method is active at a time and update dynamically."""
        if self.sing_thres_checkbox.isChecked():
            self.pro_cont_checkbox.setChecked(False)
            self.current_control_method = "single_threshold"
        elif self.pro_cont_checkbox.isChecked():
            self.sing_thres_checkbox.setChecked(False)
            self.current_control_method = "proportional_control"
        
        print(f"Switched to: {self.current_control_method}")  # Debugging feedback
        
    def add_new_option(self):
        """Add a new option to the dropdown and initialize its RMS values list"""
        text, ok = QInputDialog.getText(self, "Add New Option", "Enter new option:")
        if ok and text.strip():
            option_name = text.strip()
            # Check if option already exists
            if self.control_dropdown.findText(option_name) == -1:
                self.control_dropdown.addItem(option_name)
                # Initialize empty list for the new option
                self.option_rms_values[option_name] = []
                self.option_selected_channels[option_name] = 0
                self.option_mvc_values[option_name] = [0] * 8  # Initialize with 8 zeros
                print(f"Added new option: {option_name}")
                # Select the newly added option
                self.control_dropdown.setCurrentText(option_name)
            else:
                print(f"Option '{option_name}' already exists")
                
    def add_new_option_motor(self):
        """Add a new option to the dropdown and initialize its RMS values list"""
        text, ok = QInputDialog.getText(self, "Add New Option", "Enter new option:")
        if ok and text.strip():
            option_name = text.strip()
            # Check if option already exists
            if self.motor_dropdown.findText(option_name) == -1:
                self.motor_dropdown.addItem(option_name)
                print(f"Added new option: {option_name}")
                # Select the newly added option
                self.motor_dropdown.setCurrentText(option_name)
            else:
                print(f"Option '{option_name}' already exists")

    def on_dropdown_changed(self, text):
        """Handle dropdown selection change"""
        selected_option = text
        print(f"Selected option: {selected_option}")
        
        # Update the channel index for processing based on the selected option
        if selected_option in self.option_selected_channels:
            self.ind_chan = self.option_selected_channels[selected_option]
            # Update the threshold calculator with the new channel
            self.threshold_calculator.ind_chan = self.ind_chan
            print(f"Switched to channel {self.ind_chan+1} for {selected_option}")
            self.selected_channel_label.setText(f"Selected Channel: {self.ind_chan+1}")
        
        # Update the RMS display with the values for this option
        if selected_option in self.option_rms_values:
            rms_values = self.option_rms_values[selected_option]
            if rms_values:
                rms_text = ", ".join(str(val) for val in rms_values)
                self.rms_display.setText(f"RMS values for {selected_option}:\n{rms_text}")
            else:
                self.rms_display.setText(f"No RMS values stored for {selected_option} yet")
        
        # Update the MVC display with the values for this option
        if selected_option in self.option_mvc_values:
            mvc_values = self.option_mvc_values[selected_option]
            if any(mvc_values):  # If any MVC values are non-zero
                mvc_text = ", ".join(f"{val:.2f}" for val in mvc_values)
                self.mvc_display.setText(f"MVC values for {selected_option}:\n{mvc_text}")
            else:
                self.mvc_display.setText(f"No MVC values stored for {selected_option} yet")
            
            # Update the threshold calculator with the action-specific MVC values
            self.action_mvc_peak = np.array(mvc_values)
            self.threshold_calculator.mvc_peak = self.action_mvc_peak

                
    def update_parameters(self):
        try:
            # Convert the text inputs to appropriate numeric types
            self.current_percentage = int(self.percentage_input.text())
            self.current_angle_tolerance = float(self.update_diffangle.text())
            self.current_time_tolerance = float(self.update_difftime.text())
            print(f"Parameters updated: Percentage: {self.current_percentage}, "
                f"Angle Tolerance: {self.current_angle_tolerance}, "
                f"Time Tolerance: {self.current_time_tolerance}")
        except ValueError as e:
            print(f"Error updating parameters: {e}")
            
    def update_stimulus(self, text, bg_color, text_color):
        self.stimulus_display.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; font-size: 50px;")
        self.stimulus_display.setText(text)
        
    def segment_buffer(self):
        """Segment the buffer into overlapping windows, apply Min-Max normalization."""
        all_windows = []
    
        self.window_segments = [[] for _ in range(8)]
        self.normalized_buffers = [[] for _ in range(8)]
        processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        self.smoothed_buffers = processor.preproc_loop_buffer(self.window_size)
        
        # Extract Baseline and MVC values as lists
        baseline_values = list(map(float, self.Baseline_input.text().strip("[]").split()))
        
        # Get current action and its MVC values
        current_option = self.control_dropdown.currentText()
        if current_option in self.option_mvc_values and any(self.option_mvc_values[current_option]):
            # Use action-specific MVC if available
            mvc_values = self.option_mvc_values[current_option]

        # else:
        #     # Fall back to global MVC
        #     mvc_values = list(map(float, self.mvc_input.text().strip("[]").split()))
        #     print("Using global MVC values")

        for channel in range(self.sensor.num_channels):
            smoothed_buffer = np.array(self.smoothed_buffers[channel])

            # Apply Min-Max Normalization: (X - baseline) / (MVC - baseline)
            denominator = mvc_values[channel] - baseline_values[channel]

            if denominator == 0:  
                self.normalized_buffer = smoothed_buffer  
            else:
                self.normalized_buffer = (smoothed_buffer - baseline_values[channel]) / max(denominator, 1e-6)
                self.normalized_buffer = np.clip(self.normalized_buffer, 0, 1)
                
            self.normalized_buffers[channel] = self.normalized_buffer # Store normalized data

            if len(self.normalized_buffer) >= self.window_size:
                for i in range(0, len(self.normalized_buffer) - self.window_size + 1, self.step_size):
                    window = self.normalized_buffer[i:i + self.window_size]
                    self.window_segments[channel].append(window)
                    
                    all_windows.append({
                        "channel": channel,
                        "start_index": i,
                        "window": window
                    })

        return all_windows

    def update_plot(self):
        """Update the graph with new data and segmented windows."""
        # Update segmented windows and normalized buffers
        all_windows = self.segment_buffer()
        num_windows = min(len(self.window_segments[ch]) for ch in range(self.sensor.num_channels))
        
        for i in range(num_windows):
            # Create a batch: for each channel, get the i-th window directly.
            window_batch = [self.window_segments[ch][i] for ch in range(self.sensor.num_channels)]
            
            # [1] Single Threshold Implementation
            if self.current_control_method == "single_threshold":
                movement_status = self.threshold_calculator.Thresh_Implement(window_batch)
                self.window.set_movement_status(movement_status)
                self.window.update()
            
            # [2] Proportional Control Implementation
            elif self.current_control_method == "proportional_control":
                movement_status = self.threshold_calculator.Proportional_Implement_v5(window_batch, ind_chan=self.ind_chan, command = self.motor_dropdown.currentText(), Force_channel=self.force_channel_dropdown.currentText())

        # Now update the plot for each channel with the normalized data
        for channel in range(self.sensor.num_channels):
            smoothed_buffer = self.normalized_buffers[channel]
            x_vals = range(len(smoothed_buffer))
            y_vals = smoothed_buffer
            self.data_lines[channel].setData(x_vals, y_vals)
            if len(x_vals) > self.window_size:
                self.graph_widgets[channel].setXRange(x_vals[-self.window_size], x_vals[-1])
            else:
                self.graph_widgets[channel].setXRange(0, self.window_size)

    def on_calibrate_click(self):
        """Start baseline calibration when the button is clicked."""
        self.update_stimulus("Stay still for baseline!", "white", "black")
        print("Starting baseline calibration...")
        self.sensor.do_baseline = True

        # Start the sensor thread and timer
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.start()
        self.timer.start(20)

        # Wait for sufficient data
        while not all(len(self.sensor.baseline[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
            print("Waiting for sufficient baseline data...")
            time.sleep(0.1) 
            
        print("Performing calibration using discarded data...")
        baseline_data = np.array([list(self.sensor.baseline[ch])[:2500] for ch in range(self.sensor.num_channels)])
        processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        
        for ch in range(self.sensor.num_channels):
            preprocessed_data = processor.preprocess_filters(baseline_data[ch])
            channel_data = preprocessed_data[0:2500]
            self.mean_5s[ch] = np.mean(channel_data)
            self.std_5s[ch] = np.std(channel_data)
            
        self.Baseline_input.setText(f"{self.mean_5s}")
        print(f"Baseline Mean: {self.mean_5s}")
        print(f"Baseline Std: {self.std_5s}")
        
        # Stop the sensor thread after calibration
        self.sensor_thread.stop()
        self.timer.stop()
        
        self.sensor.do_baseline = False
        
        self.update_stimulus("Baseline Calibrated! Now calibrate MVC.", "yellow", "black")
        
        for ch in range(self.sensor.num_channels):
            self.sensor.baseline[ch] = []
            
    def on_calibrate_action_mvc_click(self):
        """Calibrate MVC specifically for the current action"""
        current_option = self.control_dropdown.currentText()
        self.update_stimulus(f"Performing MVC calibration for {current_option}...", "magenta", "white")
        print(f"Starting MVC calibration for {current_option}...")
        self.sensor.do_mvc = True

        # Start the sensor thread and timer
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.start()
        self.timer.start(20)

        # Wait for sufficient data
        while not all(len(self.sensor.mvc[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
            print("Waiting for sufficient MVC data...")
            time.sleep(0.1)

        print(f"Processing MVC data for {current_option}...")
        mvc_data = np.array([list(self.sensor.mvc[ch])[:2500] for ch in range(self.sensor.num_channels)])

        # Apply signal processing
        processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        
        # Compute MVC peak amplitudes
        action_mvc = np.zeros(8)
        for ch in range(self.sensor.num_channels):
            preprocessed_data = processor.preprocess_filters(mvc_data[ch])
            channel_data = preprocessed_data[0:2500]
            channel_data = processor.moving_window_rms(channel_data, 200)
            if len(channel_data) > 0:
                peaks, _ = find_peaks(channel_data, height=0)
                action_mvc[ch] = np.max(channel_data[peaks]) if len(peaks) > 0 else 0

        # Store the MVC values for this action
        self.option_mvc_values[current_option] = action_mvc.tolist()
        
        # Update the action MVC display
        mvc_text = ", ".join(f"{val:.2f}" for val in action_mvc)
        self.mvc_display.setText(f"MVC values for {current_option}:\n{mvc_text}")
        
        # Update the threshold calculator with the action-specific MVC values
        self.action_mvc_peak = action_mvc
        self.threshold_calculator.mvc_peak = self.action_mvc_peak
        
        print(f"MVC Peak Amplitudes for {current_option}: {action_mvc}")

        # Stop the sensor thread and clear data
        self.sensor_thread.stop()
        self.timer.stop()
        self.sensor.do_mvc = False

        # Clear MVC data after calibration
        for ch in range(self.sensor.num_channels):
            self.sensor.mvc[ch] = []

        self.update_stimulus(f"MVC Calibration for {current_option} Complete!", "green", "black")
    
    # def use_global_mvc(self):
    #     """Use the global MVC values for the current action"""
    #     current_option = self.control_dropdown.currentText()
        
    #     # Copy global MVC values to the current action
    #     self.option_mvc_values[current_option] = self.mvc_peak.tolist()
    #     self.action_mvc_peak = np.array(self.mvc_peak)
    #     self.threshold_calculator.mvc_peak = self.action_mvc_peak
        
    #     # Update the display
    #     mvc_text = ", ".join(f"{val:.2f}" for val in self.mvc_peak)
    #     self.mvc_display.setText(f"Using global MVC values for {current_option}:\n{mvc_text}")
    #     self.update_stimulus(f"Global MVC applied to {current_option}", "cyan", "black")
    #     print(f"Applied global MVC values to {current_option}")

    # def on_calibrate_mvc_click(self):
    #     """Start MVC calibration when the button is clicked."""
    #     self.update_stimulus("Performing MVC calibration...", "orange", "black")
    #     print("Starting MVC calibration...")
    #     self.sensor.do_mvc = True

    #     # Start the sensor thread and timer
    #     self.sensor_thread = SensorThread(self.sensor)
    #     self.sensor_thread.start()
    #     self.timer.start(20)

    #     # Wait for sufficient data
    #     while not all(len(self.sensor.mvc[ch]) >= 2500 for ch in range(self.sensor.num_channels)):
    #         print("Waiting for sufficient MVC data...")
    #         time.sleep(0.1)

    #     print("Processing MVC data...")
    #     mvc_data = np.array([list(self.sensor.mvc[ch])[:2500] for ch in range(self.sensor.num_channels)])

    #     # Apply signal processing
    #     processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        
    #     # Compute MVC peak amplitudes
    #     for ch in range(self.sensor.num_channels):
    #         preprocessed_data = processor.preprocess_filters(mvc_data[ch])
    #         channel_data = preprocessed_data[0:2500]
    #         channel_data = processor.moving_window_rms(channel_data, 200)
    #         if len(channel_data) > 0:
    #             peaks, _ = find_peaks(channel_data, height=0)
    #             self.mvc_peak[ch] = np.max(channel_data[peaks]) if len(peaks) > 0 else 0

    #     self.mvc_input.setText(f"{self.mvc_peak}")
    #     print(f"MVC Peak Amplitudes: {self.mvc_peak}")

    #     # Stop the sensor thread and clear data
    #     self.sensor_thread.stop()
    #     self.timer.stop()
    #     self.sensor.do_mvc = False

    #     # Clear MVC data after calibration
    #     for ch in range(self.sensor.num_channels):
    #         self.sensor.mvc[ch] = []

    #     self.update_stimulus("MVC Calibration Complete!", "green", "black")

    def on_start_click(self):
        """Start the sensor stream for real-time monitoring."""
        self.sensor.collect_data = True
        self.update_stimulus("Go!", "green", "black")
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.start()
        self.timer.start(20)

    def on_stop_click(self):
        """Stop the sensor stream."""
        self.sensor.collect_data = False
        self.update_stimulus("Stay still for baseline!", "white", "black")
        self.sensor_thread.stop()
        self.timer.stop()

    def smart_select_bf(self):
        """Start the sensor stream for real-time monitoring."""
        self.sensor.collect_data = True
        current_option = self.control_dropdown.currentText()
        self.update_stimulus(f"Smart Select for {current_option}...", "green", "black")
        
        # Get values from textboxes with error handling
        try:
            count = int(self.count_input.text())
            rest_duration = float(self.rest_duration_input.text())
            up_duration = float(self.up_duration_input.text())
            hold_duration = float(self.hold_duration_input.text())
            down_duration = float(self.down_duration_input.text())
        except ValueError:
            # If invalid input, use default values
            count = 10
            rest_duration = up_duration = hold_duration = down_duration = 0.5
            self.count_input.setText(str(count))
            self.rest_duration_input.setText(str(rest_duration))
            self.up_duration_input.setText(str(up_duration))
            self.hold_duration_input.setText(str(hold_duration))
            self.down_duration_input.setText(str(down_duration))
            self.update_stimulus("Invalid input, using defaults", "red", "black")
        
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.start()
        self.timer.start(20)
        
        # If there's an existing thread, stop it
        if self.stimulus_thread and self.stimulus_thread.isRunning():
            self.stimulus_thread.stop()
        
        # Create and start a new stimulus thread with user parameters
        self.stimulus_thread = StimulusThread(
            count=count, 
            rest_duration=rest_duration,
            up_duration=up_duration,
            hold_duration=hold_duration,
            down_duration=down_duration
        )
        self.stimulus_thread.update_stim_signal.connect(self.update_stimulus)
        self.stimulus_thread.start()

    def smart_select_save_bf(self):
        """Save RMS values and select optimal channel for current option"""
        self.sensor.collect_data = False
        self.sensor_thread.stop()
        self.timer.stop()
        
        # Get currently selected option from dropdown
        current_option = self.control_dropdown.currentText()
        
        if self.save_checkbox.isChecked():
            print(f"Processing Smart Select data for {current_option}...")
            filtered_data_df, rms_values = self.sensor.get_NRO_elements()
            rms_values_float = [float(val) for val in rms_values]
            
            # Find max RMS channel
            if rms_values_float:
                Max_element = max(rms_values_float)
                best_channel = rms_values_float.index(Max_element)
                
                # Store the selected channel for this option
                self.option_selected_channels[current_option] = best_channel
                self.ind_chan = best_channel  # Update current channel index
                
                # Update the threshold calculator with the new channel
                self.threshold_calculator.ind_chan = self.ind_chan
                
                print(f"Best channel for {current_option}: Channel {best_channel+1} (RMS: {Max_element})")
                self.update_stimulus(f"{current_option}: Channel {best_channel+1}", "orange", "black")
                self.selected_channel_label.setText(f"Selected Channel: {best_channel+1}")
            else:
                print("No valid RMS values found")
                self.update_stimulus("No valid RMS values found", "red", "white")
            
            # Store RMS values for the selected option
            self.option_rms_values[current_option] = rms_values_float
            
            # Update the RMS display
            rms_values_str = ", ".join(str(val) for val in rms_values_float)
            self.rms_display.setText(f"RMS values for {current_option}:\n{rms_values_str}")
            
            # Save to file if needed
            if not filtered_data_df.empty:
                filename = self.filename_input.text()
                suffix = self.suffix_spinbox.value()
                full_filename = f"Smart_select_{current_option}_{filename}_{suffix}.csv"
                filepath = os.path.join(os.getcwd(), full_filename)
                print(f"Saving data to: {filepath}")
                try:
                    filtered_data_df.to_csv(filepath, index=False)
                    print(f"Filtered data saved to {filepath}")
                    # Clear discarded elements after saving
                    self.sensor.clear_discarded_elements()
                except Exception as e:
                    print(f"Error saving data: {e}")
        else:
            print("No data to save - checkbox not checked.")
        
    def on_save_click(self):
        """Save the data to a file."""
        print("Save button clicked!")
        
        if self.save_checkbox.isChecked():
            print("Discarded Elements DataFrame:")
            discarded_data_df = self.sensor.get_discarded_elements()
            if not discarded_data_df.empty:
                filename = self.filename_input.text()
                suffix = self.suffix_spinbox.value()
                full_filename = f"{filename}_{suffix}.csv"
                filepath = os.path.join(os.getcwd(), full_filename)
                print(f"Saving data to: {filepath}")
                try:
                    discarded_data_df.to_csv(filepath, index=False)
                    print(f"Discarded data successfully saved to {filepath}.")
                    # Clear discarded elements after saving
                    self.sensor.clear_discarded_elements()
                except Exception as e:
                    print(f"An error occurred while saving the discarded data: {e}")
        else:
            print("No discarded data to save.")
            
    def get_current_rms_values(self):
        """Get RMS values for the currently selected option"""
        current_option = self.control_dropdown.currentText()
        return self.option_rms_values.get(current_option, [])
        
    # Add method to export all RMS values to a JSON file
    def export_all_rms_values(self):
        """Export all stored RMS values to a JSON file"""
        import json
        filename = f"{self.filename_input.text()}_rms_values.json"
        filepath = os.path.join(os.getcwd(), filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.option_rms_values, f, indent=4)
            print(f"All RMS values exported to {filepath}")
            return True
        except Exception as e:
            print(f"Error exporting RMS values: {e}")
            return False
        
    def export_all_mvc_values(self):
        """Export all stored MVC values to a JSON file"""
        import json
        filename = f"{self.filename_input.text()}_mvc_values.json"
        filepath = os.path.join(os.getcwd(), filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.option_mvc_values, f, indent=4)
            print(f"All MVC values exported to {filepath}")
            return True
        except Exception as e:
            print(f"Error exporting MVC values: {e}")
            return False
        
    def handle_file_status_update(self):
        """Handle updates to the filename, suffix, or save status."""
        filename = self.filename_input.text()
        suffix = self.suffix_spinbox.value()
        save_status = self.save_checkbox.isChecked()
        print(f"Filename: {filename}, Suffix: {suffix}, Save Status: {save_status}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    sensor = MindRoveSensor(buffer_size=300, num_channels=8)
    Cursor = GraphWidget()
    window = MyApp(sensor=sensor, window=Cursor)
    window.show()
    Cursor.show()
    sys.exit(app.exec_())