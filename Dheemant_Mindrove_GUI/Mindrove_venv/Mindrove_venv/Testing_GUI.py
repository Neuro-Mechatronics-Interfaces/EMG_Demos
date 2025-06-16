import os
import sys  
import time
import random
from datetime import datetime
import math
import threading
import pandas as pd
import numpy as np
from collections import deque
from PyQt5.QtGui import QPolygonF, QBrush, QColor
from PyQt5.QtCore import Qt, QPointF, QThread, pyqtSignal, QMutex, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QGridLayout, QGraphicsScene, QGraphicsProxyWidget,
    QPushButton, QGraphicsPolygonItem, QVBoxLayout, QHBoxLayout, QCheckBox, QSpinBox, QLabel, QLineEdit, QTextEdit, QGraphicsView, QComboBox, QInputDialog
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
import joblib
from Snake_Game import Game
from game_selection_dialog import GameSelectionDialog

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

class RealTimeStimulusThread(QThread):
    update_stim_signal = pyqtSignal(str, str, str)        # for GUI display
    update_stim_label = pyqtSignal(str, str)              # for data logging

    def __init__(self, gesture_list, pause_duration=1.0, up=0.5, hold=0.5, down=0.5, total_count=20):
        super().__init__()
        self.gesture_list = gesture_list
        self.pause_duration = pause_duration
        self.up_duration = up
        self.hold_duration = hold
        self.down_duration = down
        self.total_count = total_count
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        count = 0
        while count < self.total_count and self.running:
            gesture = random.choice(self.gesture_list)

            # Rest / pause
            self.update_stim_signal.emit("Stay Still", "white", "black")
            self.update_stim_label.emit("rest", gesture)
            self.sleep_for(self.pause_duration)

            # Ramp Up
            self.update_stim_signal.emit(f"Ramp Up: {gesture}", "white", "black")
            self.update_stim_label.emit("up", gesture)
            self.sleep_for(self.up_duration)

            # Hold
            self.update_stim_signal.emit(f"Hold: {gesture}", "white", "black")
            self.update_stim_label.emit("hold", gesture)
            self.sleep_for(self.hold_duration)

            # Ramp Down
            self.update_stim_signal.emit(f"Ramp Down: {gesture}", "white", "black")
            self.update_stim_label.emit("down", gesture)
            self.sleep_for(self.down_duration)

            count += 1

        self.update_stim_signal.emit("Prediction Complete!", "green", "black")
        self.update_stim_label.emit("complete", "None")

    def sleep_for(self, seconds):
        ms = int(seconds * 1000)
        for _ in range(ms // 10):
            if not self.running:
                break
            self.msleep(10)

        
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
        
        # --- Basic Setup ---
        self.sensor = sensor
        self.window = window
        self.window_size = window_size
        self.step_size = step_size
        self.stimulus_thread = None
        self.prediction_thread = None
        self.sensor_thread = None
        
        # --- UNIFIED TIMER ---
        # REMOVED the old 'self.timer'. This is now the only timer for updates.
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(50)  # ~20 FPS
        self.update_timer.timeout.connect(self.update_plot)

        # --- NON-BLOCKING CALIBRATION TIMER ---
        # This timer is required to prevent the GUI from freezing during calibration.
        self.calibration_check_timer = QTimer(self)
        self.calibration_check_timer.setInterval(200) # Check every 200ms
        self.calibration_check_timer.timeout.connect(self.check_calibration_status)
        self.is_calibrating = None # Will hold 'baseline' or 'mvc'
        
        # --- UI Setup ---
        self.setupUi() # Organized UI creation into a helper method

        # --- Initialize App State & Controls ---
        self.action_mvc_peak = np.zeros(8)
        self.mean_5s = np.zeros(8)
        self.std_5s = np.zeros(8)
        self.update_parameters()
        self.ind_chan = 1
        self.threshold_calculator = Controls_st_pc(
            self.sensor, self.window_size, self.step_size, self.mvc_peak, 
            self.ind_chan, self.current_percentage, 
            self.current_angle_tolerance, self.current_time_tolerance, self
        )
        self.connect_signals()
        self.on_dropdown_changed(self.control_dropdown.currentText())

    def setupUi(self):
        """Creates and arranges all UI widgets."""
        self.setWindowTitle("NML-MindRove RTMC-GUI")
        self.setGeometry(200, 200, 3000, 1500)
        self.setStyleSheet("background-color: black; color: white;")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # (The rest of your UI setup code from __init__ goes here, unchanged)
        # For example:
        self.left_layout = QVBoxLayout()
        # ... create all your buttons, inputs, etc. ...
        self.right_layout = QVBoxLayout()
        # ... setup circular graph layout ...
        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

    def connect_signals(self):
        """Connects UI element signals to their functions. No duplicates."""
        self.calibrate_baseline_button.clicked.connect(self.on_calibrate_baseline_click)
        self.calibrate_action_mvc_button.clicked.connect(self.on_calibrate_action_mvc_click)
        self.start_button.clicked.connect(self.on_start_click)
        self.stop_button.clicked.connect(self.on_stop_click)
        # ... connect all other signals once ...
        self.control_dropdown.currentTextChanged.connect(self.on_dropdown_changed)
        self.sing_thres_checkbox.toggled.connect(self.update_control_selection)
        # etc.

    ### --------------------------------------------------
    ###  CORE FUNCTIONALITY (START, STOP, UPDATE)
    ### --------------------------------------------------

    def on_start_click(self):
        """Starts the sensor and the single update timer."""
        if self.sensor_thread and self.sensor_thread.is_alive():
            return # Already running
        self.sensor.collect_data = True
        self.update_stimulus("Go!", "green", "black")
        self.sensor_thread = threading.Thread(target=self.sensor.start, daemon=True)
        self.sensor_thread.start()
        self.update_timer.start() # Starts the single timer

    def on_stop_click(self):
        """Stops the sensor and the single update timer."""
        self.update_timer.stop() # Stops the single timer
        self.sensor.collect_data = False
        if self.sensor_thread:
            self.sensor.stop()
            self.sensor_thread.join(timeout=2.0)
            self.sensor_thread = None
        self.update_stimulus("Stopped", "red", "white")

    def update_plot(self):
        """This is the main update loop called by the single QTimer."""
        # 1. Update Plots and Rotation
        for channel in range(8):
            buffer = self.sensor.get_buffers_stream(channel)
            if buffer:
                self.data_lines[channel].setData(y=list(buffer))
        
        orientation = self.sensor.get_wrist_orientation()
        if orientation is not None and len(orientation) > 0:
            self.update_graph_rotation(-np.rad2deg(orientation[0]))

        # 2. Run Control Logic
        # (Your existing logic for segmentation and prediction goes here)
        # self.segment_buffer()
        # ...
            
    ### --------------------------------------------------
    ###  NON-BLOCKING CALIBRATION (FIXES THE FREEZING)
    ### --------------------------------------------------

    def on_calibrate_baseline_click(self):
        """CHANGED: Starts baseline calibration without blocking."""
        if self.is_calibrating: return
        self.is_calibrating = "baseline"
        self.update_stimulus("Calibrating Baseline...", "white", "black")
        self.sensor.do_baseline = True
        self.sensor_thread = threading.Thread(target=self.sensor.start, daemon=True)
        self.sensor_thread.start()
        self.calibration_check_timer.start()

    def on_calibrate_action_mvc_click(self):
        """CHANGED: Starts MVC calibration without blocking."""
        if self.is_calibrating: return
        self.is_calibrating = "mvc"
        self.update_stimulus(f"Calibrating MVC...", "magenta", "white")
        self.sensor.do_mvc = True
        self.sensor_thread = threading.Thread(target=self.sensor.start, daemon=True)
        self.sensor_thread.start()
        self.calibration_check_timer.start()

    def check_calibration_status(self):
        """NEW: Helper function called by a timer to check if calibration is done."""
        if self.is_calibrating == "baseline" and all(len(d) >= self.sensor.len_baseline_data for d in self.sensor.baseline):
            print("Baseline data collected. Processing...")
            self.finish_calibration()
            # Your baseline data processing logic here...
            self.update_stimulus("Baseline Calibrated!", "green", "black")

        elif self.is_calibrating == "mvc" and all(len(d) >= self.sensor.len_mvc_data for d in self.sensor.mvc):
            print("MVC data collected. Processing...")
            self.finish_calibration()
            # Your MVC data processing logic here...
            self.update_stimulus("MVC Calibrated!", "green", "black")

    def finish_calibration(self):
        """Helper to stop threads and timers after calibration."""
        self.calibration_check_timer.stop()
        self.sensor.stop()
        if self.sensor_thread:
            self.sensor_thread.join(timeout=2.0)
        self.sensor_thread = None
        self.is_calibrating = None
        self.sensor.do_baseline = False
        self.sensor.do_mvc = False


    def update_control_selection(self):
        """Ensure only one control method is active at a time and update dynamically."""
        if self.sing_thres_checkbox.isChecked():
            self.pro_cont_checkbox.setChecked(False)
            self.cnn_class_checkbox.setChecked(False)
            self.mlp_class_checkbox.setChecked(False)
            self.current_control_method = "single_threshold"
        elif self.pro_cont_checkbox.isChecked():
            self.sing_thres_checkbox.setChecked(False)
            self.cnn_class_checkbox.setChecked(False)
            self.mlp_class_checkbox.setChecked(False)
            self.current_control_method = "proportional_control"
        elif self.cnn_class_checkbox.isChecked():
            self.sing_thres_checkbox.setChecked(False)
            self.pro_cont_checkbox.setChecked(False)
            self.mlp_class_checkbox.setChecked(False)
            self.current_control_method = "CNN_Classifier" 
        elif self.mlp_class_checkbox.isChecked():
            self.sing_thres_checkbox.setChecked(False)
            self.pro_cont_checkbox.setChecked(False)
            self.cnn_class_checkbox.setChecked(False)
            self.current_control_method = "MLP_Classifier"

        
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

    def show_game_selection_dialog(self):
        dialog = GameSelectionDialog(self, direction_queue=self.shared_direction_queue)
        dialog.exec_()
        self.snake_game_dialog = dialog  # Store reference

            
    def update_stimulus(self, text, bg_color, text_color):
        self.stimulus_display.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; font-size: 50px;")
        self.stimulus_display.setText(text)
        
    def segment_buffer(self):
        """Segment the buffer into overlapping windows, with and without Min-Max normalization."""
        self.window_segments_norm = [[] for _ in range(8)]
        self.window_segments_smoothed = [[] for _ in range(8)]
        self.window_segments_filtered = [[] for _ in range(8)]
        self.window_segments_pre_rectification = [[] for _ in range(8)]

        self.normalized_buffers = [[] for _ in range(8)]

        processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        self.smoothed_buffers, self.filtered_buffers, self.smoothed_buffers_stream, self.pre_rectification_buffers = processor.preproc_loop_buffer(self.window_size)


        # Extract Baseline and MVC values
        baseline_values = list(map(float, self.Baseline_input.text().strip("[]").split()))
        current_option = self.control_dropdown.currentText()

        if current_option in self.option_mvc_values and any(self.option_mvc_values[current_option]):
            mvc_values = self.option_mvc_values[current_option]
        else:
            mvc_values = list(map(float, self.mvc_input.text().strip("[]").split()))

        for channel in range(self.sensor.num_channels):
            smoothed_buffer = np.array(self.smoothed_buffers[channel])
            filtered_buffer = np.array(self.filtered_buffers[channel])
            pre_rectified_buffer = np.array(self.pre_rectification_buffers[channel])

            # --- Create Smoothed (RMS) windows for RMS values ---
            if len(smoothed_buffer) >= self.window_size:
                for i in range(0, len(smoothed_buffer) - self.window_size + 1, self.step_size):
                    raw_window = smoothed_buffer[i:i + self.window_size]
                    self.window_segments_smoothed[channel].append(raw_window)

            # --- Create filtered windows for (MAV) ---
            if len(filtered_buffer) >= self.window_size:
                for i in range(0, len(filtered_buffer) - self.window_size + 1, self.step_size):
                    filt_window = filtered_buffer[i:i + self.window_size]
                    self.window_segments_filtered[channel].append(filt_window)

            # --- Create prerectified windows for (ZC) ---
            if len(pre_rectified_buffer) >= self.window_size:
                for i in range(0, len(pre_rectified_buffer) - self.window_size + 1, self.step_size):
                    pre_rec_window = pre_rectified_buffer[i:i + self.window_size]
                    self.window_segments_pre_rectification[channel].append(pre_rec_window)

            # --- Create Normalize buffer ---
            denominator = mvc_values[channel] - baseline_values[channel]
            if denominator == 0:
                normalized_buffer = smoothed_buffer
            else:
                normalized_buffer = (smoothed_buffer - baseline_values[channel]) / max(denominator, 1e-6)
                normalized_buffer = np.clip(normalized_buffer, 0, 1)

            self.normalized_buffers[channel] = normalized_buffer

            # Store normalized windows
            if len(normalized_buffer) >= self.window_size:
                for i in range(0, len(normalized_buffer) - self.window_size + 1, self.step_size):
                    norm_window = normalized_buffer[i:i + self.window_size]
                    self.window_segments_norm[channel].append(norm_window)



    def update_plot(self):
        """Update the graph with new data and segmented windows."""

        # --- Part 1: Plotting and Rotation ---
        # Update EMG data on each plot
        for channel in range(8):
            buffer = self.sensor.get_buffers_stream(channel)
            if buffer:
                self.data_lines[channel].setData(y=list(buffer))
                self.graph_widgets[channel].setXRange(0, 2500) # Ensure X-range is consistent

        # Get wrist orientation and rotate graphs
        orientation = self.sensor.get_wrist_orientation()
        if orientation is not None and len(orientation) > 0:
            roll_radians = orientation[0]
            rotation_angle_degrees = -np.rad2deg(roll_radians)
            self.update_graph_rotation(rotation_angle_degrees)
            
        # --- Part 2: Control Logic (from your old update_plot method) ---
        self.segment_buffer()
        num_windows = min(len(self.window_segments_smoothed[ch]) for ch in range(self.sensor.num_channels))

        for i in range(num_windows):
            # Choose input window type depending on control method
            if self.current_control_method in ["single_threshold", "proportional_control"]:
                # Use normalized
                window_batch = [self.window_segments_norm[ch][i] for ch in range(self.sensor.num_channels)]
            elif self.current_control_method in ["CNN_Classifier", "MLP_Classifier"]:
                # Use smoothed but unnormalized
                window_batch = [self.window_segments_smoothed[ch][i] for ch in range(self.sensor.num_channels)]
            else:
                window_batch = None  # Or raise warning

            if window_batch is None:
                continue  # skip this iteration if control method is not recognized

            # --- Threshold-based control ---
            if self.current_control_method == "single_threshold":
                movement_status = self.threshold_calculator.Thresh_Implement(window_batch)
                self.window.set_movement_status(movement_status)
                self.window.update()

            # --- Proportional control ---
            elif self.current_control_method == "proportional_control":
                movement_status = self.threshold_calculator.Proportional_Implement_v5(
                    window_batch,
                    ind_chan=self.ind_chan,
                    command=self.motor_dropdown.currentText(),
                    Force_channel=self.force_channel_dropdown.currentText()
                )

            # --- CNN Classifier ---
            elif self.current_control_method == "CNN_Classifier":
                batch_array = np.array(window_batch).T  # Shape (200, 8)
                input_tensor = np.expand_dims(batch_array, axis=0)  # Shape (1, 200, 8)
                # Load model if not already loaded
                if not hasattr(self, "cnn_model") or self.cnn_model is None:
                    try:
                        self.cnn_model = joblib.load("CNN_Classifier_95_data_2025_05_16_0.pkl")  # fallback if not trained yet
                        print("Default CNN model loaded.")
                    except Exception as e:
                        print(f"Error loading default model: {e}")
                        return
                start_time = time.time()  # Start timing
                y_pred_probs = self.cnn_model.predict(input_tensor)
                y_pred = np.argmax(y_pred_probs, axis=1)[0]
                predicted_label = self.class_labels[y_pred]
                end_time = time.time()    # End timing


                # Create label columns for this window
                label_column = np.full((200, 1), predicted_label)
                gesture_column = np.full((200, 1), self.current_gesture)
                stimulus_column = np.full((200, 1), self.current_stim_state)

                # Combine window + labels → shape (200, 11)
                labeled_window = np.hstack((batch_array, label_column, gesture_column, stimulus_column))

                # Store it
                if not hasattr(self, "full_window_log"):
                    self.full_window_log = []

                self.full_window_log.append(labeled_window)

                inference_time_ms = (end_time - start_time) * 1000  # in milliseconds
                
                # print(f"Predicted Class: {y_pred} → {predicted_label}")

                if predicted_label != self.last_prediction:
                    print(f"Predicted: {predicted_label} | True: {self.current_gesture} ({self.current_stim_state}) | Inference Time: {inference_time_ms:.2f} ms")

                    display_text = (
                        f"Predicted: {predicted_label.upper()}\n"
                        f"True Gesture: {self.current_gesture.upper()} ({self.current_stim_state})"
                    )
                    self.predicted_display.setText(display_text)

                    self.last_prediction = predicted_label

            # --- MLP Classifier ---
            elif self.current_control_method == "MLP_Classifier":
                # Batch from RMS-smoothed windows (for RMS feature)
                rms_window_batch = [self.window_segments_smoothed[ch][i] for ch in range(self.sensor.num_channels)]
                rms_window_array = np.array(rms_window_batch) # shape: (8, 200)

                # Batch from filtered (bandpass+notch+rectified) windows (MAV features)
                filtered_window_batch = [self.window_segments_filtered[ch][i] for ch in range(self.sensor.num_channels)]
                filtered_window_array = np.array(filtered_window_batch) # shape: (8, 200)

                # Batch from prerectified for ZC 
                pre_rectification_window_batch = [self.window_segments_pre_rectification[ch][i] for ch in range(self.sensor.num_channels)]
                pre_rectification_window_array = np.array(pre_rectification_window_batch) # shape: (8, 200)
            
            
                # Compute RMS across time
                rms_features = np.sqrt(np.mean(np.square(rms_window_array), axis=1)).reshape(1, -1)

                #-----------------------
                # MAV and ZC are calculated on the filtered signal and the prerectified signal
                mav_features = np.mean(np.abs(filtered_window_array), axis=1).reshape(1, -1)
                zc_features = (np.sum(np.abs(np.diff(np.sign(pre_rectification_window_array), axis=1)), axis = 1) / 2).reshape(1, -1)
                
                combined_features = np.concatenate([rms_features, mav_features, zc_features], axis=1)
                #-----------------------

                # Load model if not already loaded
                if not hasattr(self, "mlp_model") or self.mlp_model is None:
                    try:
                        # self.mlp_model = joblib.load("MLP_Classifier_94_data_2025_05_16_0.pkl")  # fallback if not trained yet Flex exten
                        # self.mlp_model = joblib.load("MLP_96_2025_06_03_Dheemant_Actual_FLEX_EXTEN_RADIAL_ULNAR_0.pkl") # Snake Game
                        self.mlp_model = joblib.load("E:\Mindrove_venv\Mindrove_venv\MLP_94_2025_06_12_data_2025_06_12_Dheemant_MAV_ZC_PCA_V1_0_FLEX_EXTEN_INDEX_0.pkl") # Spacecraft Game
                        self.pca_model = joblib.load("E:\Mindrove_venv\Mindrove_venv\PCA_FLEX_EXTEN_INDEX_data_2025_06_12_Dheemant_MAV_ZC_PCA_V1_0.pkl")
                        print("Default MLP model loaded.")
                    except Exception as e:
                        print(f"Error loading default model: {e}")
                        return

                features_pca = self.pca_model.transform(combined_features)
                start_time = time.time()  # Start timing
                #-----------------------
                y_pred = self.mlp_model.predict(features_pca)
                #-----------------------
                # y_pred = self.mlp_model.predict(rms_features)
                end_time = time.time()    # End timing

                predicted_label = y_pred[0]

                
                # # Store 200 labels for the window
                # if not hasattr(self, "Label_whole_data"):
                #     self.Label_whole_data = []

                # label_column = np.full((200, 1), predicted_label)
                # self.Label_whole_data.append(label_column)

                window_array = np.array(rms_window_batch).T  # shape: (200, 8)

                # Create label columns for this window
                label_column = np.full((200, 1), predicted_label)
                gesture_column = np.full((200, 1), self.current_gesture)
                stimulus_column = np.full((200, 1), self.current_stim_state)

                # Combine window + labels → shape (200, 11)
                labeled_window = np.hstack((window_array, label_column, gesture_column, stimulus_column))

                # Store it
                if not hasattr(self, "full_window_log"):
                    self.full_window_log = []

                self.full_window_log.append(labeled_window)

                inference_time_ms = (end_time - start_time) * 1000  # in milliseconds

                if predicted_label != self.last_prediction:
                    print(f"Predicted: {predicted_label} | True: {self.current_gesture} ({self.current_stim_state}) | Inference Time: {inference_time_ms:.2f} ms")

                    display_text = (
                        f"Predicted: {predicted_label}\n"
                        f"True Gesture: {self.current_gesture} ({self.current_stim_state})"
                    )
                    self.predicted_display.setText(display_text)

                    # Send control to Snake game
                    if hasattr(self, "snake_game_dialog") and hasattr(self.snake_game_dialog, "direction_queue"):
                        print(predicted_label)
                        self.snake_game_dialog.direction_queue.put(predicted_label)

                    # Send control to Space Shooter game
                    if hasattr(self, "SSG_game_dialog") and hasattr(self.SSG_game_dialog, "direction_queue"):
                        print(predicted_label)
                        self.SSG_game_dialog.direction_queue.put(predicted_label)


                    self.last_prediction = predicted_label


        # --- Plot buffer_stream data ---
        for channel in range(self.sensor.num_channels):
            smoothed_buffers_stream_chan = self.smoothed_buffers_stream[channel]
            x_vals = range(len(smoothed_buffers_stream_chan))
            y_vals = smoothed_buffers_stream_chan
            self.data_lines[channel].setData(x_vals, y_vals)

            
            self.graph_widgets[channel].setXRange(0, 2500)

    def on_train_click(self):
        """Train classifier using multiple gesture files."""
        from LDA_Train_Class import LDA_Trainer_Multi

        filename = self.filename_input.text()
        suffix = self.suffix_spinbox.value()
        base_path = os.getcwd()

        gesture_filepaths = {}

        # Gather all available gesture files from dropdown list
        for i in range(self.control_dropdown.count()):
            gesture = self.control_dropdown.itemText(i)
            filepath = os.path.join(base_path, f"Smart_select_{gesture}_{filename}_{suffix}_unfilt.csv")
            if os.path.exists(filepath):
                gesture_filepaths[gesture] = filepath
            else:
                print(f"[Warning] File not found for gesture: {gesture} at {filepath}")

        if not gesture_filepaths:
            self.update_stimulus("No valid gesture files found", "red", "white")
            return

        try:
            trainer = LDA_Trainer_Multi(
                gesture_filepaths=gesture_filepaths,
                file_identifier=filename,
                file_suffix=suffix,
                classifier_type=self.current_control_method
            )
            accuracy = trainer.train_model()
            self.mlp_model = trainer.mlp_model
            self.latest_model_path = trainer.model_save_path

            self.update_stimulus(f"Training Complete! Accuracy: {accuracy*100:.2f}%", "cyan", "black")
            print(f"[INFO] Trained and switched to: {self.latest_model_path}")

        except Exception as e:
            self.update_stimulus("Training Failed", "red", "white")
            print(f"[ERROR] Training error: {e}")


    def start_prediction_prompt(self):
        if self.prediction_thread and self.prediction_thread.isRunning():
            self.prediction_thread.stop()
            self.prediction_thread.wait()

        gestures = [self.control_dropdown.itemText(i) for i in range(self.control_dropdown.count())]
        pause = float(self.rest_duration_input.text())
        up = float(self.up_duration_input.text())
        hold = float(self.hold_duration_input.text())
        down = float(self.down_duration_input.text())

        self.prediction_thread = RealTimeStimulusThread(
            gesture_list=gestures,
            pause_duration=pause,
            up=up,
            hold=hold,
            down=down,
            total_count=int(self.count_input.text())
        )

        self.prediction_thread.update_stim_signal.connect(self.update_stimulus)
        self.prediction_thread.update_stim_label.connect(self.capture_stimulus_label)
        self.prediction_thread.start()

    def capture_stimulus_label(self, stim_state, gesture):
        # If rest phase → force gesture to "rest"
        if stim_state == "rest":
            self.current_stim_state = "rest"
            self.current_gesture = "rest"
        else:
            self.current_stim_state = stim_state
            self.current_gesture = gesture if gesture is not None else "rest"

    def update_graph_rotation(self, rotation_angle_deg):
        center_x, center_y = 750, 750
        radius = 500
        graph_width, graph_height = 300, 200
        for i, (graph_proxy, label_proxy) in enumerate(zip(self.graph_proxies, self.label_proxies)):
            rotated_angle_deg = self.initial_angles_deg[i] + rotation_angle_deg
            rotated_angle_rad = math.radians(rotated_angle_deg)
            x = center_x + radius * math.cos(rotated_angle_rad) - (graph_width / 2)
            y = center_y + radius * math.sin(rotated_angle_rad) - (graph_height / 2)
            graph_proxy.setPos(x, y)
            label_x = x + 40
            label_y = y - 10
            label_proxy.setPos(label_x, label_y)

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
            unfiltered_data_df, filtered_data_df, rms_values = self.sensor.get_NRO_elements()
            rms_values_float = [float(val) for val in rms_values]
            
            # Add stimulus labels to the filtered data
            if not (filtered_data_df.empty or unfiltered_data_df.empty):
                # Create label column if it doesn't exist
                if 'stimulus_state' not in filtered_data_df.columns:
                    filtered_data_df['stimulus_state'] = None
                    unfiltered_data_df['stimulus_state'] = None
                
                # Get timing parameters from inputs
                rest_duration = float(self.rest_duration_input.text())
                up_duration = float(self.up_duration_input.text())
                hold_duration = float(self.hold_duration_input.text())
                down_duration = float(self.down_duration_input.text())
                count = int(self.count_input.text())
                
                # Calculate total cycle duration
                cycle_duration = rest_duration + up_duration + hold_duration + down_duration
                
                # Calculate labels for each sample based on time
                sampling_freq = 500  # Hz
                total_samples = len(filtered_data_df)
                
                for i in range(total_samples):
                    # Calculate elapsed time for this sample
                    elapsed_time = i / sampling_freq
                    
                    # Determine which repetition this sample belongs to
                    rep_number = int(elapsed_time / cycle_duration)
                    
                    # If we've exceeded the requested count, all remaining samples are "rest"
                    if rep_number >= count:
                        filtered_data_df.at[i, 'stimulus_state'] = "rest"
                        unfiltered_data_df.at[i, 'stimulus_state'] = "rest"
                        continue
                    
                    # Calculate time within the current cycle
                    time_in_cycle = elapsed_time % cycle_duration
                    
                    # Determine stimulus state based on time in cycle
                    if time_in_cycle < rest_duration:
                        filtered_data_df.at[i, 'stimulus_state'] = "rest"
                        unfiltered_data_df.at[i, 'stimulus_state'] = "rest"
                    elif time_in_cycle < rest_duration + up_duration:
                        filtered_data_df.at[i, 'stimulus_state'] = "up"
                        unfiltered_data_df.at[i, 'stimulus_state'] = "up"
                    elif time_in_cycle < rest_duration + up_duration + hold_duration:
                        filtered_data_df.at[i, 'stimulus_state'] = "hold"
                        unfiltered_data_df.at[i, 'stimulus_state'] = "hold"
                    else:
                        filtered_data_df.at[i, 'stimulus_state'] = "down"
                        unfiltered_data_df.at[i, 'stimulus_state'] = "down"
                
                print(f"Added stimulus state labels to {total_samples} samples")
            
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
                full_filename = f"Smart_select_{current_option}_{filename}_{suffix}_filt.csv"
                full_filename_uf = f"Smart_select_{current_option}_{filename}_{suffix}_unfilt.csv"
                filepath = os.path.join(os.getcwd(), full_filename)
                filepath_uf = os.path.join(os.getcwd(), full_filename_uf)
                print(f"Saving data to: {filepath}")
                try:
                    filtered_data_df.to_csv(filepath, index=False)
                    unfiltered_data_df.to_csv(filepath_uf, index=False)
                    print(f"Filtered and Unfiltered data saved to {filepath} and {filepath_uf}")
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
                    
            if hasattr(self, "full_window_log") and self.full_window_log:
                for i in range(len(self.full_window_log)):
                    window = self.full_window_log[i]
                    
                    # Check if gesture or stimulus columns (last two columns) have None
                    if window.shape[1] >= 10:
                        gesture_col = window[:, -2]
                        stimulus_col = window[:, -1]

                        gesture_col[gesture_col == None] = "rest"
                        stimulus_col[stimulus_col == None] = "rest"

                        # Replace in the window
                        self.full_window_log[i][:, -2] = gesture_col
                        self.full_window_log[i][:, -1] = stimulus_col

                # Stack all windows: shape (num_windows * 200, 9)
                full_data_array = np.vstack(self.full_window_log)

                # Create DataFrame and save
                col_names = [f"Ch{i+1}" for i in range(8)] + ["Label", "Gesture", "Stimulus"]
                full_window_df = pd.DataFrame(full_data_array, columns=col_names)


                labeled_filename = f"{filename}_{suffix}_full_window_log.csv"
                labeled_filepath = os.path.join(os.getcwd(), labeled_filename)

                full_window_df.to_csv(labeled_filepath, index=False)
                print(f"Full window data saved to {labeled_filepath}")

                self.full_window_log.clear()

        else:
            print("No discarded data to save.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    from Single_thresh_Control import initialize_arduino
    initialize_arduino()  

    app = QApplication(sys.argv)
    sensor = MindRoveSensor(buffer_size=300, num_channels=8)
    Cursor = GraphWidget()

    window = MyApp(sensor=sensor, window=Cursor)
    window.show()
    Cursor.show()
    
    sys.exit(app.exec_())
