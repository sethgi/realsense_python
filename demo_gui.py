#!/usr/bin/env python3
"""
RealSense RGBD + IMU Viewer
Lightweight GUI optimized for Raspberry Pi
"""

import sys
import numpy as np
from collections import deque
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGroupBox, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

import rs_python


class IMUWidget(QWidget):
    """Lightweight IMU visualization widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Data storage
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        
        # History for plotting (last 100 samples)
        self.history_len = 100
        self.accel_history = deque(maxlen=self.history_len)
        self.gyro_history = deque(maxlen=self.history_len)
        
        # Initialize with zeros
        for _ in range(self.history_len):
            self.accel_history.append(np.zeros(3))
            self.gyro_history.append(np.zeros(3))
        
        # Colors for X, Y, Z
        self.colors = [QColor(255, 80, 80), QColor(80, 255, 80), QColor(80, 80, 255)]
        self.axis_labels = ['X', 'Y', 'Z']
    
    def update_data(self, accel, gyro):
        self.accel = accel
        self.gyro = gyro
        self.accel_history.append(accel.copy())
        self.gyro_history.append(gyro.copy())
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        half_h = h // 2
        
        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        # Draw accel graph (top half)
        self._draw_graph(painter, 0, 0, w, half_h - 5, 
                        self.accel_history, "Accel (m/s²)", 20.0)
        
        # Draw gyro graph (bottom half)
        self._draw_graph(painter, 0, half_h + 5, w, half_h - 5,
                        self.gyro_history, "Gyro (rad/s)", 5.0)
    
    def _draw_graph(self, painter, x, y, w, h, history, title, scale):
        # Border
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawRect(x, y, w - 1, h - 1)
        
        # Title
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Sans", 9, QFont.Bold))
        painter.drawText(x + 5, y + 14, title)
        
        # Center line
        center_y = y + h // 2
        painter.setPen(QPen(QColor(60, 60, 60), 1, Qt.DashLine))
        painter.drawLine(x, center_y, x + w, center_y)
        
        # Plot data
        margin = 20
        plot_h = (h - margin) // 2
        data = np.array(list(history))
        
        for axis in range(3):
            painter.setPen(QPen(self.colors[axis], 1))
            
            points = []
            for i, val in enumerate(data[:, axis]):
                px = x + int(i * (w - 1) / self.history_len)
                # Clamp and scale
                normalized = np.clip(val / scale, -1, 1)
                py = center_y - int(normalized * plot_h)
                points.append((px, py))
            
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1],
                               points[i+1][0], points[i+1][1])
        
        # Current values
        painter.setFont(QFont("Sans", 8))
        current = data[-1] if len(data) > 0 else np.zeros(3)
        for i, (val, color) in enumerate(zip(current, self.colors)):
            painter.setPen(color)
            text = f"{self.axis_labels[i]}:{val:+6.2f}"
            painter.drawText(x + w - 70, y + 14 + i * 12, text)


class OrientationWidget(QWidget):
    """Simple 3D orientation cube visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Euler angles (simple integration, will drift)
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.last_time = time.time()
    
    def update_gyro(self, gyro, dt):
        # Simple integration (for visualization only - will drift!)
        self.roll += gyro[0] * dt
        self.pitch += gyro[1] * dt
        self.yaw += gyro[2] * dt
        
        # Wrap angles
        self.roll = self.roll % (2 * np.pi)
        self.pitch = self.pitch % (2 * np.pi)
        self.yaw = self.yaw % (2 * np.pi)
        
        self.update()
    
    def reset(self):
        self.roll = self.pitch = self.yaw = 0
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        size = min(w, h) // 3
        
        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))
        
        # Title
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Sans", 9, QFont.Bold))
        painter.drawText(5, 14, "Orientation")
        
        # Draw simple horizon indicator
        # Roll indicator (outer ring)
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawEllipse(cx - size, cy - size, size * 2, size * 2)
        
        # Horizon line (affected by roll)
        roll_deg = np.degrees(self.roll)
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        dx = int(size * 0.8 * np.cos(self.roll))
        dy = int(size * 0.8 * np.sin(self.roll))
        painter.drawLine(cx - dx, cy - dy, cx + dx, cy + dy)
        
        # Pitch indicator (vertical bar on side)
        pitch_y = int(np.clip(self.pitch / (np.pi/4), -1, 1) * (size * 0.8))
        painter.setPen(QPen(QColor(200, 200, 0), 2))
        painter.drawLine(cx, cy, cx, cy - pitch_y)
        painter.drawEllipse(cx - 3, cy - pitch_y - 3, 6, 6)
        
        # Center marker
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(cx - 10, cy, cx + 10, cy)
        painter.drawLine(cx, cy - 10, cx, cy + 10)
        
        # Angle readouts
        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Sans", 8))
        painter.drawText(5, h - 35, f"R: {np.degrees(self.roll):+6.1f}°")
        painter.drawText(5, h - 22, f"P: {np.degrees(self.pitch):+6.1f}°")
        painter.drawText(5, h - 9, f"Y: {np.degrees(self.yaw):+6.1f}°")


class ImageWidget(QLabel):
    """Efficient image display widget"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #404040;")
        self._pixmap = None
    
    def update_image(self, image, is_depth=False):
        if is_depth:
            # Colorize depth: normalize and apply colormap
            depth_display = self._colorize_depth(image)
            h, w = depth_display.shape[:2]
            bytes_per_line = w * 3
            qimg = QImage(depth_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = image.shape[:2]
            bytes_per_line = w * 3
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget while maintaining aspect ratio
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_scaled()
    
    def _colorize_depth(self, depth):
        """Apply a simple colormap to depth image"""
        # Normalize to 0-255 (assuming max useful depth ~5m = 5000mm)
        depth_normalized = np.clip(depth.astype(np.float32) / 5000.0, 0, 1)
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        
        # Simple turbo-like colormap
        r = np.clip(1.5 - np.abs(depth_normalized * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(depth_normalized * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(depth_normalized * 4 - 1), 0, 1)
        
        colored = np.stack([
            (r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8)
        ], axis=-1)
        
        # Mark invalid depth (0) as black
        colored[depth == 0] = 0
        
        return colored
    
    def _update_scaled(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
            self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense RGBD + IMU Viewer")
        self.setMinimumSize(800, 500)
        
        # Initialize camera
        self.cam = None
        self.init_camera()
        
        # Setup UI
        self.setup_ui()
        
        # FPS tracking
        self.frame_count = 0
        self.fps_time = time.time()
        self.fps = 0
        self.last_imu_time = time.time()
        
        # Update timer (target ~30fps for Pi)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
    
    def init_camera(self):
        try:
            self.cam = rs_python.RSCam(enable_imu=True)
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            self.cam = None
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        
        # Left side: RGB and Depth images
        image_layout = QVBoxLayout()
        
        # RGB
        rgb_group = QGroupBox("RGB")
        rgb_layout = QVBoxLayout(rgb_group)
        rgb_layout.setContentsMargins(2, 2, 2, 2)
        self.rgb_widget = ImageWidget("RGB")
        rgb_layout.addWidget(self.rgb_widget)
        image_layout.addWidget(rgb_group)
        
        # Depth
        depth_group = QGroupBox("Depth")
        depth_layout = QVBoxLayout(depth_group)
        depth_layout.setContentsMargins(2, 2, 2, 2)
        self.depth_widget = ImageWidget("Depth")
        depth_layout.addWidget(self.depth_widget)
        image_layout.addWidget(depth_group)
        
        main_layout.addLayout(image_layout, stretch=2)
        
        # Right side: IMU
        imu_layout = QVBoxLayout()
        
        # IMU graphs
        imu_group = QGroupBox("IMU Data")
        imu_group_layout = QVBoxLayout(imu_group)
        self.imu_widget = IMUWidget()
        imu_group_layout.addWidget(self.imu_widget)
        imu_layout.addWidget(imu_group, stretch=2)
        
        # Orientation
        orient_group = QGroupBox("Orientation (integrated)")
        orient_layout = QVBoxLayout(orient_group)
        self.orient_widget = OrientationWidget()
        orient_layout.addWidget(self.orient_widget)
        imu_layout.addWidget(orient_group, stretch=1)
        
        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #aaa; font-size: 10px;")
        imu_layout.addWidget(self.status_label)
        
        main_layout.addLayout(imu_layout, stretch=1)
        
        # Styling
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox { 
                color: #ddd; 
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
    
    def update_frame(self):
        if not self.cam:
            self.status_label.setText("Camera not connected")
            return
        
        try:
            # Get RGBD
            rgb, depth = self.cam.GetRGBD()
            self.rgb_widget.update_image(rgb)
            self.depth_widget.update_image(depth, is_depth=True)
            
            # Get IMU
            if self.cam.IsIMUEnabled():
                imu = self.cam.GetIMU()
                accel = imu['accel']
                gyro = imu['gyro']
                
                self.imu_widget.update_data(accel, gyro)
                
                # Update orientation
                current_time = time.time()
                dt = current_time - self.last_imu_time
                self.last_imu_time = current_time
                self.orient_widget.update_gyro(gyro, dt)
            
            # FPS calculation
            self.frame_count += 1
            elapsed = time.time() - self.fps_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_time = time.time()
            
            self.status_label.setText(f"FPS: {self.fps:.1f} | IMU: {'ON' if self.cam.IsIMUEnabled() else 'OFF'}")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            # Reset orientation
            self.orient_widget.reset()
        elif event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            self.close()
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.cam:
            self.cam.Stop()
        event.accept()


def main():
    # For Raspberry Pi: use software rendering if needed
    # import os
    # os.environ['QT_QUICK_BACKEND'] = 'software'
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Consistent look across platforms
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
