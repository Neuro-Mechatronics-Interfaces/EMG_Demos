#------------------------------------------------

# Completed & Works
# 3/21/25

#------------------------------------------------
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
        
        self.setStyleSheet("background-color: black; color: white;")
        self.sensor = sensor
        self.processor = SignalProcessor(self.sensor, fs=500, highpass_cutoff=10, notch_freq=60, notch_q=30)
        self.window_size = window_size
        self.step_size = step_size
        self.window = window
        self.stimulus_thread = None
        self.setWindowTitle("NML-MindRove RTMC-GUI")
        self.setGeometry(200, 200, 3000, 1500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # # Create and configure a QLabel to hold the background image
        # self.bg_image_label = QLabel(self.central_widget)
        # self.bg_image_label.setAlignment(Qt.AlignLeft| Qt.AlignBottom)

        # image = QPixmap("E:/Mindrove_venv/Mindrove_venv/NML_logo.png")

        # transparent_image = QPixmap(image.size())
        # transparent_image.fill(Qt.transparent)
        # painter = QPainter(transparent_image)
        # painter.setOpacity(0.5)  
        # painter.drawPixmap(0, 0, image)
        # painter.end()
        
        # # Set the image to the label
        # self.bg_image_label.setPixmap(transparent_image)
        # self.bg_image_label.setGeometry(600, 1350, image.width(), image.height())
        
        
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
        self.cnn_class_checkbox = QCheckBox("CNN Classifier")
        self.mlp_class_checkbox = QCheckBox("MLP Classifier")

        # Ensure only one can be selected at a time
        self.sing_thres_checkbox.toggled.connect(self.update_control_selection)
        self.pro_cont_checkbox.toggled.connect(self.update_control_selection)
        self.cnn_class_checkbox.toggled.connect(self.update_control_selection)
        self.mlp_class_checkbox.toggled.connect(self.update_control_selection)

        checkbox_layout.addWidget(self.sing_thres_checkbox)
        checkbox_layout.addWidget(self.pro_cont_checkbox)
        checkbox_layout.addWidget(self.cnn_class_checkbox)
        checkbox_layout.addWidget(self.mlp_class_checkbox)
        
        # Dictionary to store RMS values for each dropdown option
        self.option_rms_values = {}
        # Dictionary to store the selected channel index for each option
        self.option_selected_channels = {}
        self.option_mvc_values = {}
        
        self.control_dropdown = QComboBox()
        # Initialize with default options and their corresponding blank lists
        default_options = ["Wflex", "WExten"]
        self.class_labels = ["rest", "wexten", "wflex"]
        
        self.last_prediction = None # Being used in MLP
        self.pca_model = None
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
        self.cnn_class_checkbox.setChecked(False)
        self.mlp_class_checkbox.setChecked(False)

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
        self.train_button = QPushButton('Train', self)
        self.prediction_stim_button = QPushButton("Start Prompting", self)
        self.calibrate_orientation_button = QPushButton("Calibrate Zero Position")

        self.games_button = QPushButton("Games", self) # The new "Games" button
        self.shared_direction_queue = multiprocessing.Queue()
  

        self.calibrate_baseline_button.setStyleSheet("background-color: yellow; color: black;")
        self.calibrate_action_mvc_button.setStyleSheet("background-color: orange; color: black;")
        self.start_button.setStyleSheet("background-color: green; color: black;")
        self.stop_button.setStyleSheet("background-color: red; color: black;")
        self.save_button.setStyleSheet("background-color: yellow; color: black;")
        self.smart_select.setStyleSheet("background-color: yellow; color: black;")
        self.smart_select_save.setStyleSheet("background-color: yellow; color: black;")
        self.train_button.setStyleSheet("background-color: cyan; color: black;")
        self.prediction_stim_button.setStyleSheet("background-color: pink; color: black;")
        self.games_button.setStyleSheet("background-color: pink; color: black;")
        self.calibrate_orientation_button.setStyleSheet("background-color: yellow; color: black;")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.calibrate_baseline_button)
        button_layout.addWidget(self.calibrate_action_mvc_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.smart_select)
        button_layout.addWidget(self.smart_select_save)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.prediction_stim_button)
        button_layout.addWidget(self.games_button)
        button_layout.addWidget(self.calibrate_orientation_button)
        
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

        # predicted Label
        self.predicted_display_label = QLabel("Predicted Label:")
        self.left_layout.addWidget(self.predicted_display_label)

        self.predicted_display = QTextEdit()
        self.predicted_display.setReadOnly(True)
        self.predicted_display.setAlignment(Qt.AlignCenter)
        self.predicted_display.setMaximumHeight(60)
        self.left_layout.addWidget(self.predicted_display)

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

        # # Normal Vertical Graphing
        # channel_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'white']

        # self.graph_widgets = []
        # self.data_lines = []

        # for channel in range(8):
        #     graph_widget = pg.PlotWidget()

        #     # Set no axes, ticks, or borders
        #     # graph_widget.hideAxis('left')
        #     # graph_widget.hideAxis('bottom')
        #     graph_widget.setMenuEnabled(False)
        #     graph_widget.setMouseEnabled(x=False, y=False)
        #     graph_widget.setFrameStyle(0)

        #     graph_widget.setBackground('k')  # black background 
        #     graph_widget.setLimits(xMin=0)
        #     graph_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        #     graph_widget.enableAutoRange(x=False, y=False)
        #     graph_widget.setYRange(0, 200.0)  
        #     graph_widget.setXRange(0, 1000)

        #     self.graph_widgets.append(graph_widget)
        #     self.right_layout.addWidget(graph_widget)

        #     # Create plot line with different color
        #     pen = pg.mkPen(color=channel_colors[channel % len(channel_colors)], width=2)
        #     data_line = graph_widget.plot(pen=pen)
        #     self.data_lines.append(data_line)
        # --------------------------------------------------------------------------------------
        # Create container widget for circular layout
        # self.circular_widget = QWidget()
        # self.circular_layout = QGraphicsScene()
        # self.circular_view = QGraphicsView(self.circular_layout, self.circular_widget)

        # self.circular_view.setStyleSheet("background-color: black; border: none;")
        # self.circular_view.setFixedSize(1500, 1500)  # Adjust to fit nicely

        # self.right_layout.addWidget(self.circular_view)

        # channel_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'white']
        # self.graph_widgets = []
        # self.data_lines = []

        # # Circle parameters
        # center_x, center_y = 320, 320  # Center of circular view
        # radius = 500  # Radius of the circular layout

        # # Clock positions 
        # clock_angles_deg = [337.5+5, 292.5+5, 247.5-5, 202.5-5, 157.5+5, 112.5+5, 22.5-5, 67.5-5]  # Channels 0–7
        # clock_angles_rad = [math.radians(deg) for deg in clock_angles_deg]

        # for channel in range(8):
        #     graph_widget = pg.PlotWidget()
        #     graph_widget.setMenuEnabled(False)
        #     graph_widget.setMouseEnabled(x=False, y=False)
        #     graph_widget.setFrameStyle(0)
        #     graph_widget.setBackground('k')
        #     graph_widget.setYRange(0, 200.0)
        #     graph_widget.setXRange(0, 2500)
        #     graph_widget.setFixedSize(300, 200)

        #     pen = pg.mkPen(color=channel_colors[channel % len(channel_colors)], width=2)
        #     data_line = graph_widget.plot(pen=pen)

        #     self.graph_widgets.append(graph_widget)
        #     self.data_lines.append(data_line)

        #     # Polar coordinates to screen position
        #     angle = clock_angles_rad[channel]
        #     x = center_x + radius * math.cos(angle) - 60  # subtract half width
        #     y = center_y + radius * math.sin(angle) - 40  # subtract half height

        #     proxy = QGraphicsProxyWidget()
        #     proxy.setWidget(graph_widget)
        #     proxy.setPos(x, y)
        #     self.circular_layout.addItem(proxy)
        #     label = QLabel(f"Ch {channel+1}")
        #     label.setStyleSheet("color: white; background-color: transparent;")
        #     proxy_label = QGraphicsProxyWidget()
        #     proxy_label.setWidget(label)
        #     proxy_label.setPos(x + 40, y - 10)  # Adjust label offset as needed
        #     self.circular_layout.addItem(proxy_label)
        # -----------------------------------------------------

        self.circular_widget = QWidget()
        self.circular_layout = QGraphicsScene()
        self.circular_view = QGraphicsView(self.circular_layout, self.circular_widget)

        self.circular_view.setStyleSheet("background-color: black; border: none;")
        self.circular_view.setFixedSize(1500, 1500)  

        self.right_layout.addWidget(self.circular_view)
        
        # --- ADDED FOR ROTATION ---
        # Make these instance variables to access them in the update method
        self.center_x, self.center_y = 750, 750  # Center of the QGraphicsView
        self.radius = 500  # Radius of the circular layout
        # --- END ADDITION ---

        logo_label = QLabel()
        image = QPixmap("E:/Mindrove_venv/Mindrove_venv/NML_Logo_.png")
        image = image.scaledToWidth(200, Qt.SmoothTransformation)
        painter = QPainter(image)
        painter.setOpacity(0.3)  
        painter.drawPixmap(0, 0, image)
        painter.end()
        logo_label.setPixmap(image)

        logo_label.setFixedSize(image.size())

        logo_proxy = QGraphicsProxyWidget()
        logo_proxy.setWidget(logo_label)
        logo_x = self.center_x - logo_label.width() / 2
        logo_y = self.center_y - logo_label.height() / 2
        logo_proxy.setPos(logo_x, logo_y)
        logo_proxy.setZValue(-1)

        self.circular_layout.addItem(logo_proxy)


        channel_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'white']
        self.graph_widgets = []
        self.data_lines = []
        
        # --- ADDED FOR ROTATION ---
        # Store proxies to update their positions later
        self.graph_proxies = []
        self.label_proxies = []
        self.rotation_offset = 0.0
        # Store the initial angles to use as a base for rotation
        clock_angles_deg = [337.5+5, 292.5+5, 247.5-5, 202.5-5, 157.5+5, 112.5+5, 22.5-5, 67.5-5]
        
        self.initial_angles_rad = [math.radians(deg) for deg in clock_angles_deg]
        # --- END ADDITION ---


        for channel in range(8):
            graph_widget = pg.PlotWidget()
            graph_widget.setMenuEnabled(False)
            graph_widget.setMouseEnabled(x=False, y=False)
            graph_widget.setFrameStyle(0)
            graph_widget.setBackground('k')
            graph_widget.setYRange(0, 200.0)
            graph_widget.setXRange(0, 2500)
            graph_widget.setFixedSize(300, 200)

            pen = pg.mkPen(color=channel_colors[channel % len(channel_colors)], width=2)
            data_line = graph_widget.plot(pen=pen)

            self.graph_widgets.append(graph_widget)
            self.data_lines.append(data_line)

            # --- MODIFIED FOR ROTATION ---
            # Initial position calculation is the same
            angle = self.initial_angles_rad[channel]
            x = self.center_x + self.radius * math.cos(angle) - graph_widget.width() / 2
            y = self.center_y + self.radius * math.sin(angle) - graph_widget.height() / 2

            proxy = QGraphicsProxyWidget()
            proxy.setWidget(graph_widget)
            proxy.setPos(x, y)
            self.circular_layout.addItem(proxy)
            self.graph_proxies.append(proxy) # Store the proxy

            label = QLabel(f"Ch {channel+1}")
            label.setStyleSheet("color: white; background-color: transparent;")
            proxy_label = QGraphicsProxyWidget()
            proxy_label.setWidget(label)
            # Position the label relative to the graph widget's corner
            proxy_label.setPos(x + 40, y - 10) 
            self.circular_layout.addItem(proxy_label)
            self.label_proxies.append(proxy_label) # Store the proxy
            # --- END MODIFICATION ---
        # --------------------------------------------------------------------------------------

        self.main_layout.addLayout(self.right_layout)

        self.calibrate_baseline_button.clicked.connect(self.on_calibrate_click)
        self.calibrate_action_mvc_button.clicked.connect(self.on_calibrate_action_mvc_click)
        self.start_button.clicked.connect(self.on_start_click)
        self.stop_button.clicked.connect(self.on_stop_click)
        self.save_button.clicked.connect(self.on_save_click)
        self.smart_select.clicked.connect(self.smart_select_bf)
        self.smart_select_save.clicked.connect(self.smart_select_save_bf)
        self.train_button.clicked.connect(self.on_train_click)
        self.prediction_stim_button.clicked.connect(self.start_prediction_prompt)
        self.games_button.clicked.connect(self.show_game_selection_dialog)
        self.calibrate_orientation_button.clicked.connect(self.on_calibrate_orientation)

        self.prediction_thread = None
        self.current_stim_state = "rest"
        self.current_gesture = self.control_dropdown.currentText()

        
        self.mean_5s = np.zeros(8)
        self.std_5s = np.zeros(8)
        self.mvc_peak = np.zeros(8)
        
        # TODO: Add separate timer for sampling (period: 20) and graphics (period: 50)
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
    
    # Add this new method to your MyApp class

    def on_calibrate_orientation(self):
        """
        Captures the current wrist rotation and sets it as the 'zero' offset.
        """
        # Instruct the user to hold their wrist in the desired zero position
        # (e.g., hand flat, palm down)
        
        current_orientation = self.sensor.get_wrist_orientation()
        if current_orientation is not None:
            self.rotation_offset = current_orientation[0] # Store the current roll in radians
            print(f"Orientation zero point calibrated. Offset = {self.rotation_offset:.2f} radians")
            self.update_stimulus("Orientation Calibrated!", "blue", "white")

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

        # TODO: This will go into its own timer loop, where you now have only periodic calls of
        #       self.processor.preproc_loop_buffer()
        self.smoothed_buffers, self.filtered_buffers, self.smoothed_buffers_stream, self.pre_rectification_buffers = self.processor.preproc_loop_buffer(self.window_size)


        # Extract Baseline and MVC values
        baseline_values = list(map(float, self.Baseline_input.text().strip("[]").split()))
        current_option = self.control_dropdown.currentText()

        if current_option in self.option_mvc_values and any(self.option_mvc_values[current_option]):
            mvc_values = self.option_mvc_values[current_option]
        else:
            mvc_values = list(map(float, self.mvc_input.text().strip("[]").split()))

        for channel in range(self.sensor.num_channels):
            # These all become references to values in self.processor, i.e.
            #   smoothed_buffer = self.processor.smoothed_buffer.get_n_recent_samples(n)
            #   filtered_buffer = self.processor.filtered_buffer.get_n_recent_samples(n)
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
        # --- ADDED FOR ROTATION (New)---
        # Call the rotation update function each time we plot
        self.update_graph_rotation()
        # --- END ADDITION ---

        # TODO: Decouple this part (sample polling) from graphics update
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
                # MAV and ZC are calculated on the filteredsignal and the prerectified signal
                mav_features = np.mean(np.abs(filtered_window_array), axis=1).reshape(1, -1)
                zc_features = (np.sum(np.abs(np.diff(np.sign(pre_rectification_window_array), axis=1)), axis = 1) / 2).reshape(1, -1)
                
                combined_features = np.concatenate([rms_features, mav_features, zc_features], axis=1)
                #-----------------------

                # Load model if not already loaded
                if not hasattr(self, "mlp_model") or self.mlp_model is None:
                    try:
                        # self.mlp_model = joblib.load("MLP_Classifier_94_data_2025_05_16_0.pkl")  # fallback if not trained yet Flex exten
                        # self.mlp_model = joblib.load("MLP_96_2025_06_03_Dheemant_Actual_FLEX_EXTEN_RADIAL_ULNAR_0.pkl") # Snake Game
                        # self.mlp_model = joblib.load("E:\Mindrove_venv\Mindrove_venv\MLP_94_2025_06_12_data_2025_06_12_Dheemant_MAV_ZC_PCA_V1_0_FLEX_EXTEN_INDEX_0.pkl") # Spacecraft Game
                        # self.pca_model = joblib.load("E:\Mindrove_venv\Mindrove_venv\PCA_FLEX_EXTEN_INDEX_data_2025_06_12_Dheemant_MAV_ZC_PCA_V1_0.pkl")

                        self.mlp_model = joblib.load("E:\Mindrove_venv\MLP_Classifier_90_data_2025_06_16_Wflex_WExten_FIndex_0.pkl") # Spacecraft Game
                        self.pca_model = joblib.load("E:\Mindrove_venv\PCA_for_MLP_Classifier_data_2025_06_16_Wflex_WExten_FIndex_0.pkl")

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

    # --- NEW METHOD FOR ROTATION (New)---
    # Replace your existing update_graph_rotation with this version

    def update_graph_rotation(self):
        """ Get wrist orientation, apply the calibration offset, and rotate the graphs. """
        orientation = self.sensor.get_wrist_orientation()
        
        # 1. APPLY THE CALIBRATION (The Key Change)
        # Subtract the stored offset from the current sensor reading.
        current_rotation = orientation[0] - self.rotation_offset

        # 2. CHOOSE THE ROTATION DIRECTION (Polarity)
        # If the rotation feels backward, change the '-' to a '+'.
        # One of these will feel correct.
        rotation_direction = -1.0 # Use -1.0 for one direction, 1.0 for the other
        
        for i in range(len(self.graph_proxies)):
            # Apply the final, calibrated rotation to the graph's initial angle
            new_angle = self.initial_angles_rad[i] + (current_rotation * rotation_direction)

            # Recalculate position (your existing code is fine here)
            graph_width = self.graph_widgets[i].width()
            graph_height = self.graph_widgets[i].height()
            
            x = self.center_x + self.radius * math.cos(new_angle) - graph_width / 2
            y = self.center_y + self.radius * math.sin(new_angle) - graph_height / 2

            self.graph_proxies[i].setPos(x, y)
            self.label_proxies[i].setPos(x + 40, y - 10)
    # --- END NEW METHOD ---


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
            self.pca_model = trainer.pca_model

            self.latest_model_path = trainer.model_save_path
            self.latest_pca_path = trainer.pca_save_path

            self.update_stimulus(f"Training Complete! Accuracy: {accuracy*100:.2f}%", "cyan", "black")
            print(f"[INFO] Trained and switched to: {self.latest_model_path}")
            print(f"[INFO] Trained and switched to: {self.latest_pca_path}")

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
            preprocessed_data = processor.rectify_emg(processor.preprocess_filters(baseline_data[ch]))
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
            preprocessed_data = processor.rectify_emg(processor.preprocess_filters(mvc_data[ch]))
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

    def on_start_click(self):
        """Start the sensor stream for real-time monitoring."""
        self.sensor.collect_data = True
        self.update_stimulus("Go!", "green", "black")
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.start()
        self.timer.start(20)

        # Default labeling before prompting begins
        self.current_stim_state = "rest"
        self.current_gesture = "rest"

        

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