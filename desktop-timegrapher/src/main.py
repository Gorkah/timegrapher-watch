#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desktop Timegrapher Application - Main Entry Point
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import QTimer
from ui.main_window import MainWindow
from audio.audio_manager import AudioManager
from analysis.timegrapher_analyzer_extreme import TimegrapherAnalyzerExtreme
from visualization.plot_manager import PlotManager

class TimegrapherApp:
    """Main application controller for the Desktop Timegrapher application"""
    
    def __init__(self):
        """Initialize the application, connect all components"""
        self.app = QApplication(sys.argv)
        self.main_window = MainWindow()
        self.audio_manager = AudioManager()
        self.analyzer = TimegrapherAnalyzerExtreme()
        self.plot_manager = PlotManager(self.main_window)
        
        # Default settings
        self.sample_rate = 44100
        self.beat_rate = 28800  # BPH (beats per hour)
        self.frame_size = 4096
        self.is_running = False
        self.update_interval = 100  # milliseconds
        
        # Connect signals and slots
        self.setup_connections()
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_analysis)
        
        # Setup microphone test timer
        self.mic_test_timer = QTimer()
        self.mic_test_timer.timeout.connect(self.update_mic_test)
        self.is_testing_mic = False
        
    def setup_connections(self):
        """Connect UI signals to their handlers"""
        # Connect UI controls
        self.main_window.start_button.clicked.connect(self.toggle_analysis)
        self.main_window.beat_rate_combo.currentIndexChanged.connect(self.set_beat_rate)
        self.main_window.mic_combo.currentIndexChanged.connect(self.select_microphone)
        self.main_window.export_button.clicked.connect(self.export_results)
        self.main_window.reset_button.clicked.connect(self.reset_analysis)
        self.main_window.test_mic_button.clicked.connect(self.toggle_mic_test)
        
        # Load microphone list
        self.load_microphones()
        
    def load_microphones(self):
        """Load available microphones into the UI"""
        mic_list = self.audio_manager.get_input_devices()
        self.main_window.mic_combo.clear()
        for mic in mic_list:
            self.main_window.mic_combo.addItem(mic['name'], mic['index'])
    
    def select_microphone(self, index):
        """Set the selected microphone as input device"""
        device_index = self.main_window.mic_combo.itemData(index)
        self.audio_manager.set_input_device(device_index)
        
    def set_beat_rate(self, index):
        """Update the beat rate based on user selection"""
        beat_rates = [18000, 21600, 28800, 36000]  # Standard watch beat rates
        if index >= 0 and index < len(beat_rates):
            self.beat_rate = beat_rates[index]
            self.analyzer.set_beat_rate(self.beat_rate)
            
            # Update UI with the new expected tick interval
            tick_ms = 3600 * 1000 / self.beat_rate
            self.main_window.status_bar.showMessage(
                f"Beat rate set to {self.beat_rate} BPH (tick every {tick_ms:.2f} ms)"
            )
    
    def toggle_analysis(self):
        """Start or stop the analysis"""
        if not self.is_running:
            # Start analysis
            try:
                self.audio_manager.start_stream(
                    self.sample_rate, 
                    self.frame_size,
                    self.audio_data_callback
                )
                self.update_timer.start(self.update_interval)
                self.is_running = True
                self.main_window.start_button.setText("Stop")
                self.main_window.status_bar.showMessage("Analysis running...")
            except Exception as e:
                QMessageBox.critical(
                    self.main_window,
                    "Error Starting Analysis",
                    f"Could not start audio stream: {str(e)}"
                )
        else:
            # Stop analysis
            self.audio_manager.stop_stream()
            self.update_timer.stop()
            self.is_running = False
            self.main_window.start_button.setText("Start")
            self.main_window.status_bar.showMessage("Analysis stopped")
    
    def audio_data_callback(self, audio_data):
        """Callback function for audio data"""
        # Process the incoming audio data
        self.analyzer.process_audio_data(audio_data, self.sample_rate)
    
    def update_analysis(self):
        """Update the analysis results in the UI"""
        # Get latest analysis results
        results = self.analyzer.get_results()
        
        # Update UI with results
        if results:
            self.main_window.update_metrics(
                rate=results.get('rate', 0),
                beat_error=results.get('beat_error', 0),
                amplitude=results.get('amplitude', 0),
                confidence=results.get('confidence', 0)
            )
            
            # Update plots
            self.plot_manager.update_plots(results)
    
    def export_results(self):
        """Export the current analysis results"""
        if not self.analyzer.has_data():
            QMessageBox.warning(
                self.main_window,
                "No Data",
                "No analysis data to export. Run the analysis first."
            )
            return
            
        # Export data to CSV
        try:
            filename = self.main_window.get_save_filename("CSV Files (*.csv)")
            if filename:
                self.analyzer.export_results(filename)
                self.main_window.status_bar.showMessage(f"Results exported to {filename}")
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Export Error",
                f"Could not export results: {str(e)}"
            )
    
    def reset_analysis(self):
        """Reset all analysis data and plots"""
        # Stop analysis if running
        if self.is_running:
            self.toggle_analysis()
            
        # Reset analyzer
        self.analyzer.reset()
        
        # Clear plots
        self.plot_manager.clear_plots()
        
        # Reset metrics display
        self.main_window.update_metrics(rate=0, beat_error=0, amplitude=0, confidence=0)
        self.main_window.status_bar.showMessage("Analysis reset")
    
    def toggle_mic_test(self):
        """Start or stop microphone test"""
        if self.is_testing_mic:
            # Stop test
            self.mic_test_timer.stop()
            self.audio_manager.stop_stream()
            self.is_testing_mic = False
            self.main_window.test_mic_button.setText("Test Microphone")
            self.main_window.status_bar.showMessage("Microphone test stopped")
        else:
            # Stop analysis if running
            if self.is_running:
                self.toggle_analysis()
                
            # Start test
            try:
                self.audio_manager.start_stream(
                    self.sample_rate,
                    self.frame_size,
                    self.mic_test_callback
                )
                self.mic_test_timer.start(100)  # Update every 100ms
                self.is_testing_mic = True
                self.main_window.test_mic_button.setText("Stop Test")
                self.main_window.status_bar.showMessage("Testing microphone...")
            except Exception as e:
                QMessageBox.critical(
                    self.main_window,
                    "Error Testing Microphone",
                    f"Could not start audio stream: {str(e)}"
                )
    
    def mic_test_callback(self, audio_data):
        """Callback for microphone test audio data"""
        # Store the audio data for visualization
        self.test_audio_data = audio_data
    
    def update_mic_test(self):
        """Update the microphone test visualization"""
        if hasattr(self, 'test_audio_data') and self.test_audio_data is not None:
            # Calculate audio level (RMS)
            rms = np.sqrt(np.mean(np.square(self.test_audio_data)))
            level_db = 20 * np.log10(max(rms, 1e-10))
            
            # Update UI to show audio level
            self.plot_manager.update_mic_test(self.test_audio_data, level_db)
    
    def run(self):
        """Run the application main loop"""
        self.main_window.show()
        return self.app.exec_()


if __name__ == "__main__":
    app = TimegrapherApp()
    sys.exit(app.run())
