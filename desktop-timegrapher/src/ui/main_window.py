#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Window UI for the Desktop Timegrapher Application
"""

import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QFileDialog,
    QFrame, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in Qt"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.fig.tight_layout()
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MainWindow(QMainWindow):
    """Main window of the Desktop Timegrapher application"""
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # Window setup
        self.setWindowTitle("Desktop Timegrapher")
        self.setMinimumSize(QSize(1000, 700))
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create UI elements
        self.create_controls()
        self.create_metrics_display()
        self.create_plot_area()
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
    def create_controls(self):
        """Create the control panel"""
        control_group = QGroupBox("Controls")
        control_layout = QGridLayout()
        
        # Beat rate selection
        beat_rate_label = QLabel("Beat Rate:")
        self.beat_rate_combo = QComboBox()
        self.beat_rate_combo.addItems(["18000 BPH", "21600 BPH", "28800 BPH", "36000 BPH"])
        self.beat_rate_combo.setCurrentIndex(2)  # Default to 28800 BPH
        
        # Microphone selection
        mic_label = QLabel("Microphone:")
        self.mic_combo = QComboBox()
        
        # Start/Stop button
        self.start_button = QPushButton("Start")
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        
        # Export button
        self.export_button = QPushButton("Export Results")
        
        # Test Microphone button
        self.test_mic_button = QPushButton("Test Microphone")
        self.test_mic_button.setToolTip("Play test sounds to verify microphone input")
        
        # Add widgets to layout
        control_layout.addWidget(beat_rate_label, 0, 0)
        control_layout.addWidget(self.beat_rate_combo, 0, 1)
        control_layout.addWidget(mic_label, 1, 0)
        control_layout.addWidget(self.mic_combo, 1, 1)
        control_layout.addWidget(self.start_button, 0, 2)
        control_layout.addWidget(self.reset_button, 0, 3)
        control_layout.addWidget(self.export_button, 1, 2)
        control_layout.addWidget(self.test_mic_button, 1, 3)
        
        control_group.setLayout(control_layout)
        self.main_layout.addWidget(control_group)
        
    def create_metrics_display(self):
        """Create the metrics display panel"""
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QHBoxLayout()
        
        # Rate display
        rate_group = QGroupBox("Rate (s/day)")
        rate_layout = QVBoxLayout()
        self.rate_value = QLabel("0.0")
        self.rate_value.setAlignment(Qt.AlignCenter)
        self.rate_value.setFont(QFont("Arial", 24, QFont.Bold))
        rate_layout.addWidget(self.rate_value)
        rate_group.setLayout(rate_layout)
        
        # Beat error display
        beat_error_group = QGroupBox("Beat Error (ms)")
        beat_error_layout = QVBoxLayout()
        self.beat_error_value = QLabel("0.0")
        self.beat_error_value.setAlignment(Qt.AlignCenter)
        self.beat_error_value.setFont(QFont("Arial", 24, QFont.Bold))
        beat_error_layout.addWidget(self.beat_error_value)
        beat_error_group.setLayout(beat_error_layout)
        
        # Amplitude display
        amplitude_group = QGroupBox("Amplitude (°)")
        amplitude_layout = QVBoxLayout()
        self.amplitude_value = QLabel("0.0")
        self.amplitude_value.setAlignment(Qt.AlignCenter)
        self.amplitude_value.setFont(QFont("Arial", 24, QFont.Bold))
        amplitude_layout.addWidget(self.amplitude_value)
        amplitude_group.setLayout(amplitude_layout)
        
        # Confidence display
        confidence_group = QGroupBox("Confianza (%)")
        confidence_layout = QVBoxLayout()
        self.confidence_value = QLabel("0.0")
        self.confidence_value.setAlignment(Qt.AlignCenter)
        self.confidence_value.setFont(QFont("Arial", 24, QFont.Bold))
        confidence_layout.addWidget(self.confidence_value)
        confidence_group.setLayout(confidence_layout)
        
        # Add metric groups to layout
        metrics_layout.addWidget(rate_group)
        metrics_layout.addWidget(beat_error_group)
        metrics_layout.addWidget(amplitude_group)
        metrics_layout.addWidget(confidence_group)
        
        metrics_group.setLayout(metrics_layout)
        self.main_layout.addWidget(metrics_group)
        
    def create_plot_area(self):
        """Create the visualization area with plots"""
        plot_group = QGroupBox("Visualization")
        plot_layout = QVBoxLayout()
        
        # Create plot canvases
        self.timing_canvas = MplCanvas(self, width=5, height=3)
        self.timing_canvas.axes.set_title("Timing Trace")
        self.timing_canvas.axes.set_xlabel("Time (s)")
        self.timing_canvas.axes.set_ylabel("Deviation (ms)")
        self.timing_canvas.axes.grid(True)
        
        self.scatter_canvas = MplCanvas(self, width=5, height=3)
        self.scatter_canvas.axes.set_title("Timegrapher Display")
        self.scatter_canvas.axes.set_xlabel("Time (ms)")
        self.scatter_canvas.axes.set_ylabel("Amplitude")
        self.scatter_canvas.axes.set_xlim(-200, 200)
        self.scatter_canvas.axes.set_ylim(-1, 1)
        self.scatter_canvas.axes.grid(True)
        
        # Add canvases to layout
        plot_layout.addWidget(self.timing_canvas)
        plot_layout.addWidget(self.scatter_canvas)
        
        plot_group.setLayout(plot_layout)
        self.main_layout.addWidget(plot_group)
        
    def update_metrics(self, rate=0, beat_error=0, amplitude=0, confidence=0):
        """Update the displayed metrics"""
        self.rate_value.setText(f"{rate:.1f}")
        # Color-code rate based on acceptable ranges
        if abs(rate) <= 5:
            self.rate_value.setStyleSheet("color: green")
        elif abs(rate) <= 15:
            self.rate_value.setStyleSheet("color: orange")
        else:
            self.rate_value.setStyleSheet("color: red")
            
        self.beat_error_value.setText(f"{beat_error:.1f}")
        # Color-code beat error based on acceptable ranges
        if beat_error <= 0.5:
            self.beat_error_value.setStyleSheet("color: green")
        elif beat_error <= 1.0:
            self.beat_error_value.setStyleSheet("color: orange")
        else:
            self.beat_error_value.setStyleSheet("color: red")
            
        self.amplitude_value.setText(f"{amplitude:.1f}")
        # Color-code amplitude based on acceptable ranges
        if amplitude >= 250:
            self.amplitude_value.setStyleSheet("color: green")
        elif amplitude >= 200:
            self.amplitude_value.setStyleSheet("color: orange")
        else:
            self.amplitude_value.setStyleSheet("color: red")
            
        self.confidence_value.setText(f"{confidence:.1f}")
        # Color-code confidence based on ranges
        if confidence >= 80:
            self.confidence_value.setStyleSheet("color: green")
        elif confidence >= 50:
            self.confidence_value.setStyleSheet("color: orange")
        else:
            self.confidence_value.setStyleSheet("color: red")
    
    def get_save_filename(self, filter_str):
        """Show save file dialog and return selected filename"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", filter_str, options=options
        )
        return file_name
