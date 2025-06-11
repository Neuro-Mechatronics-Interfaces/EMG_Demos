import sys
import math
import threading
import time
from collections import deque
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton , QGraphicsProxyWidget, QSizePolicy, QGraphicsScene, QGraphicsView
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QTimer
import pandas as pd

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
            time.sleep(0.01)  # Adjust the sleep time as needed

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

        # PyQtGraph setup for multiple channels
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create container widget for circular layout
        self.circular_widget = QWidget()
        self.circular_layout = QGraphicsScene()
        self.circular_view = QGraphicsView(self.circular_layout, self.circular_widget)

        self.circular_view.setStyleSheet("background-color: black; border: none;")
        self.circular_view.setFixedSize(1200, 1200)  # Adjust to fit nicely

        self.layout.addWidget(self.circular_view)

        channel_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'white']
        self.graph_widgets = []
        self.data_lines = []

        # Circle parameters
        center_x, center_y = 320, 320  # Center of circular view
        radius = 400  # Radius of the circular layout

        # Clock positions 
        clock_angles_deg = [337.5+5, 292.5+5, 247.5-5, 202.5-5, 157.5+5, 112.5+5, 22.5-5, 67.5-5]  # Channels 0–7
        clock_angles_rad = [math.radians(deg) for deg in clock_angles_deg]

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

            # Polar coordinates to screen position
            angle = clock_angles_rad[channel]
            x = center_x + radius * math.cos(angle) - 60  # subtract half width
            y = center_y + radius * math.sin(angle) - 40  # subtract half height

            proxy = QGraphicsProxyWidget()
            proxy.setWidget(graph_widget)
            proxy.setPos(x, y)
            self.circular_layout.addItem(proxy)
            label = QLabel(f"Ch {channel+1}")
            label.setStyleSheet("color: white; background-color: transparent;")
            proxy_label = QGraphicsProxyWidget()
            proxy_label.setWidget(label)
            proxy_label.setPos(x + 40, y - 10)  # Adjust label offset as needed
            self.circular_layout.addItem(proxy_label)

        
        self.stop_button = QPushButton("Stop")
        self.layout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.handle_stop)
        # Timer for real-time updates
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update every 100ms

        # Start sensor data collection in a separate thread
        self.sensor_thread = SensorThread(self.sensor)
        self.sensor_thread.data_updated.connect(self.update_plot)
        self.sensor_thread.start()

    def segment_buffer(self):
        """Segment the buffer into windows."""
        self.window_segments = {ch: [] for ch in range(self.sensor.num_channels)}
        for channel in range(self.sensor.num_channels):
            buffer = list(self.sensor.get_buffer(channel))
            if len(buffer) >= self.window_size:
                for i in range(0, len(buffer) - self.window_size + 1, self.step_size):
                    window = buffer[i:i + self.window_size]
                    self.window_segments[channel].append((i, window))

    def update_plot(self):
        """Update the graph with new data and segmented windows."""
        self.segment_buffer()

        # Update plots for each channel
        for channel in range(self.sensor.num_channels):
            buffer = list(self.sensor.get_buffer(channel))
            if buffer:
                print(f"Channel {channel + 1} Buffer: {buffer}")  # Print buffer contents
                for i, window in self.window_segments[channel]:
                    print(f"Channel {channel + 1} Window {i}-{i+self.window_size}: {window}")

            # Update the plot
            x_vals = range(len(buffer))
            y_vals = buffer
            self.data_lines[channel].setData(x_vals, y_vals)
            if x_vals:
                self.graph_widgets[channel].setXRange(x_vals[0], x_vals[-1])
                
    def handle_stop(self):
        """Stop the timer, save data, and exit the application."""
        print("Stop button clicked. Stopping timer and saving data...")
        self.sensor_thread.stop()
        self.timer.stop()
        print("Sensor session stopped.")
        discarded_data_df = self.sensor.get_discarded_elements()
        # discarded_data_df.to_csv("new_sp.csv", index=False)
        print("Discarded data saved to '.csv'.")
        print("Exiting application...")
        QApplication.quit()

class MockSensor:
    """Simulated sensor with 8 channels and internal buffer management."""
    def __init__(self, buffer_size=200, num_channels=8):
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.buffers = [deque(maxlen=buffer_size) for _ in range(num_channels)]
        self.running = False

    def start(self):
        """Start generating data for all channels and adding to the buffer."""
        self.running = True
        while self.running:
            for channel in range(self.num_channels):
                self.buffers[channel].extend([random.randint(1, 50) for _ in range(10)])
            time.sleep(0.01)  # Simulate data arrival every 100ms

    def get_buffer(self, channel):
        """Get the current state of the buffer for a specific channel."""
        return self.buffers[channel]

    def stop(self):
        """Stop the sensor."""
        self.running = False

    def get_discarded_elements(self):
        """Get discarded elements from the buffer."""
        discarded_data = {f"Channel {i+1}": list(self.buffers[i]) for i in range(self.num_channels)}
        return pd.DataFrame(discarded_data)


def main():
    app = QApplication(sys.argv)

    # Initialize the mock sensor with 8 channels
    sensor = MockSensor(buffer_size=200, num_channels=8)

    # Start the scrolling plot
    plot_window = ScrollingPlot(sensor=sensor, window_size=50, step_size=20)
    plot_window.show()

    # Exit handling
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()