#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Manager for the Desktop Timegrapher Application
Handles visualization of timegrapher data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle
import time

class PlotManager:
    """Manages plots for the timegrapher visualization"""
    
    def __init__(self, main_window):
        """Initialize the plot manager"""
        self.main_window = main_window
        self.timing_canvas = main_window.timing_canvas
        self.scatter_canvas = main_window.scatter_canvas
        
        # Configure plots
        self.setup_plots()
        
        # Data for plots
        self.time_data = []
        self.deviation_data = []
        self.scatter_x = []
        self.scatter_y = []
        self.scatter_sizes = []
        
        # Plot elements
        self.timing_line = None
        self.scatter_points = None
        
    def setup_plots(self):
        """Setup initial plot configurations"""
        # Configure timing trace plot
        self.timing_canvas.axes.set_title("Timing Trace")
        self.timing_canvas.axes.set_xlabel("Time (s)")
        self.timing_canvas.axes.set_ylabel("Deviation (ms)")
        self.timing_canvas.axes.grid(True)
        
        # Configure scatter plot (timegrapher display)
        self.scatter_canvas.axes.set_title("Timegrapher Display")
        self.scatter_canvas.axes.set_xlabel("Deviation (ms)")
        self.scatter_canvas.axes.set_ylabel("Position")
        self.scatter_canvas.axes.set_xlim(-5, 5)
        self.scatter_canvas.axes.set_ylim(-1.5, 1.5)
        self.scatter_canvas.axes.grid(True)
        
        # Add reference lines to scatter plot
        self.scatter_canvas.axes.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        for x in range(-5, 6):
            if x != 0:
                self.scatter_canvas.axes.axvline(x=x, color='gray', linestyle=':', alpha=0.3)
        
        # Default text for mic test mode
        self.scatter_canvas.axes.text(0, 0, "No Test Running", 
                                     ha='center', va='center', fontsize=12,
                                     color='gray', style='italic')
        
        # Update canvases
        self.timing_canvas.draw()
        self.scatter_canvas.draw()
    
    def update_plots(self, results):
        """Update plots with new data"""
        if not results:
            return
        
        # Update timing trace plot
        tick_times = results.get('tick_times', [])
        tick_intervals = results.get('tick_intervals', [])
        
        if tick_times and tick_intervals:
            # Convert to relative time and deviation
            start_time = tick_times[0] if tick_times[0] > 0 else time.time()
            
            # Prepare data for timing trace
            rel_times = [(t - start_time) for t in tick_times]
            
            # Calculate deviation from expected tick interval
            nominal_interval = 3600 * 1000 / results.get('beat_rate', 28800) / 2  # ms
            deviations = [(interval - nominal_interval) for interval in tick_intervals]
            
            # Update time series plot
            self.timing_canvas.axes.clear()
            self.timing_canvas.axes.set_title("Timing Trace")
            self.timing_canvas.axes.set_xlabel("Time (s)")
            self.timing_canvas.axes.set_ylabel("Deviation (ms)")
            self.timing_canvas.axes.grid(True)
            
            # Only plot last 200 data points for performance
            if len(rel_times) > 1 and len(deviations) > 0:
                plot_times = rel_times[-201:-1]
                plot_devs = deviations[-200:]
                
                # Make sure we have matching array lengths
                min_len = min(len(plot_times), len(plot_devs))
                if min_len > 5:  # Need enough points to make a meaningful plot
                    self.timing_canvas.axes.plot(plot_times[:min_len], plot_devs[:min_len], '-b', linewidth=1)
                    
                    # Add trend line
                    try:
                        z = np.polyfit(plot_times[:min_len], plot_devs[:min_len], 1)
                        p = np.poly1d(z)
                        self.timing_canvas.axes.plot(plot_times[:min_len], p(plot_times[:min_len]), "r--", alpha=0.8)
                    except:
                        pass  # Skip trend line if fitting fails
            
            # Update timegrapher display (scatter plot)
            tick_patterns = results.get('tick_patterns', [])
            
            if tick_patterns:
                self.scatter_canvas.axes.clear()
                
                # Redraw reference lines
                self.scatter_canvas.axes.axvline(x=0, color='r', linestyle='-', alpha=0.3)
                for x in range(-5, 6):
                    if x != 0:
                        self.scatter_canvas.axes.axvline(x=x, color='gray', linestyle=':', alpha=0.3)
                
                # Extract data for scatter plot
                x_values = [tp['x'] for tp in tick_patterns]
                y_values = [tp['y'] for tp in tick_patterns]
                amplitudes = [tp['amplitude'] * 100 + 20 for tp in tick_patterns]  # Scale for visibility
                
                # Plot scatter points
                self.scatter_canvas.axes.scatter(
                    x_values, y_values, 
                    s=amplitudes, 
                    c=y_values, cmap='coolwarm',
                    alpha=0.7
                )
                
                # Configure scatter plot
                self.scatter_canvas.axes.set_title("Timegrapher Display")
                self.scatter_canvas.axes.set_xlabel("Deviation (ms)")
                self.scatter_canvas.axes.set_ylabel("Position")
                self.scatter_canvas.axes.set_xlim(-5, 5)
                self.scatter_canvas.axes.set_ylim(-1.5, 1.5)
                self.scatter_canvas.axes.grid(True)
                
            # Refresh canvases
            self.timing_canvas.draw()
            self.scatter_canvas.draw()
    
    def update_mic_test(self, audio_data, level_db):
        """Update plots with microphone test data"""
        # Clear existing plots
        self.timing_canvas.axes.clear()
        self.scatter_canvas.axes.clear()
        
        # Setup timing plot for audio waveform
        self.timing_canvas.axes.set_title("Microphone Waveform")
        self.timing_canvas.axes.set_xlabel("Sample")
        self.timing_canvas.axes.set_ylabel("Amplitude")
        self.timing_canvas.axes.grid(True)
        
        # Plot waveform (show at most 1000 samples for performance)
        samples_to_show = min(len(audio_data), 1000)
        self.timing_canvas.axes.plot(audio_data[-samples_to_show:], 'b-', linewidth=0.5)
        self.timing_canvas.axes.set_ylim(-1, 1)
        
        # Setup scatter plot for audio level
        self.scatter_canvas.axes.set_title("Audio Level")
        self.scatter_canvas.axes.set_xlabel("Level (dB)")
        self.scatter_canvas.axes.set_ylabel("")
        self.scatter_canvas.axes.grid(True)
        
        # Set range for different audio levels
        self.scatter_canvas.axes.set_xlim(-60, 0)  # dB range
        self.scatter_canvas.axes.set_ylim(0, 1)
        
        # Add level markers
        level_colors = [
            (-60, -40, 'r', 'Too Low'),    # Red for too quiet
            (-40, -20, 'y', 'Good'),       # Yellow for good
            (-20, 0, 'g', 'Good'),         # Green for good
            (0, None, 'r', 'Too High')      # Red for too loud/clipping
        ]
        
        for start, end, color, label in level_colors:
            if end is None:  # Last segment
                rect = Rectangle((start, 0), 60, 1, color=color, alpha=0.3)
            else:
                rect = Rectangle((start, 0), end-start, 1, color=color, alpha=0.3)
            self.scatter_canvas.axes.add_patch(rect)
            
        # Add text labels for ranges
        for start, end, _, label in level_colors:
            if end is None:
                mid = start + 30
            else:
                mid = (start + end) / 2
            self.scatter_canvas.axes.text(mid, 0.5, label, ha='center', va='center')
        
        # Add a marker for current level
        self.scatter_canvas.axes.axvline(x=level_db, color='black', linestyle='-', linewidth=2)
        level_text = f"{level_db:.1f} dB"
        self.scatter_canvas.axes.text(level_db, 0.8, level_text,
                                    ha='center', va='center', 
                                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Refresh canvases
        self.timing_canvas.draw()
        self.scatter_canvas.draw()
    
    def clear_plots(self):
        """Clear all plots"""
        self.timing_canvas.axes.clear()
        self.scatter_canvas.axes.clear()
        self.setup_plots()
        
        # Clear data
        self.time_data = []
        self.deviation_data = []
        self.scatter_x = []
        self.scatter_y = []
        self.scatter_sizes = []
