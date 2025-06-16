import sys
from PyQt5.QtWidgets import QApplication, QGraphicsPolygonItem, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QPolygonF, QBrush, QColor
import pyqtgraph as pg


class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Triangle Control Example')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.graph_widget = pg.PlotWidget()
        layout.addWidget(self.graph_widget)

        self.graph_widget.setXRange(-20, 20)
        self.graph_widget.setYRange(-20, 20)

        self.triangle_pos = [0, 0]  # Initial position of the triangle
        self.triangle_upward = True  # Initial orientation of the triangle

        self.triangle_item = self.create_triangle(upward=True, color=QColor('green'), x=0, y=0)
        self.graph_widget.addItem(self.triangle_item)

        # Timer for continuous movement
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)

    def create_triangle(self, upward, color, x, y):
        size = 2  # Size of the triangle
        if upward:
            points = [
                QPointF(x, y + size),  # Top point
                QPointF(x - size, y - size),  # Bottom-left point
                QPointF(x + size, y - size),  # Bottom-right point
            ]
        else:
            points = [
                QPointF(x, y - size),  # Bottom point
                QPointF(x - size, y + size),  # Top-left point
                QPointF(x + size, y + size),  # Top-right point
            ]

        polygon = QPolygonF(points)
        triangle = QGraphicsPolygonItem(polygon)
        triangle.setBrush(QBrush(color))
        triangle.setPen(pg.mkPen(None))  # No border

        return triangle

    def update_triangle(self):
        """Update the triangle's position and orientation."""
        size = 2
        if self.triangle_upward:
            points = [
                QPointF(self.triangle_pos[0], self.triangle_pos[1] + size),
                QPointF(self.triangle_pos[0] - size, self.triangle_pos[1] - size),
                QPointF(self.triangle_pos[0] + size, self.triangle_pos[1] - size),
            ]
        else:
            points = [
                QPointF(self.triangle_pos[0], self.triangle_pos[1] - size),
                QPointF(self.triangle_pos[0] - size, self.triangle_pos[1] + size),
                QPointF(self.triangle_pos[0] + size, self.triangle_pos[1] + size),
            ]

        polygon = QPolygonF(points)
        self.triangle_item.setPolygon(polygon)
        color = QColor('green') if self.triangle_upward else QColor('red')
        self.triangle_item.setBrush(QBrush(color))

    def set_movement_status(self, movement_status):
        """Update triangle direction based on movement detection."""
        self.triangle_upward = movement_status == "Up"
        self.update_triangle()

        # Start or stop the timer based on movement status
        if movement_status == "Up":
            self.timer.start(100)  # Update every 100 ms
        else:
            self.timer.stop()

    def update_position(self):
        """Update the triangle's position continuously."""
        if self.triangle_upward:
            self.triangle_pos[1] += 0.5  # Move upwards (increase y-coordinate)
        else:
            self.triangle_pos[1] -= 0.5  # Move downwards (decrease y-coordinate)

        self.update_triangle()