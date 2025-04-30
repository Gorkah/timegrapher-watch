#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Timegrapher Analyzer for the Desktop Timegrapher Application
Performs advanced signal analysis on watch ticking sounds, optimized for faint sounds
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import peakutils
from collections import deque
import time
try:
    import pywt  # PyWavelets for wavelet denoising
except ImportError:
    print("PyWavelets not available - wavelet denoising will be skipped")
    pywt = None

class TimegrapherAnalyzer:
    """Enhanced analyzer for watch timing signals with ultra-sensitive detection"""
    
    def __init__(self):
        """Initialize the analyzer with improved sensitivity settings"""
        # Default parameters
        self.beat_rate = 28800  # Default to 28800 BPH
        self.expected_interval = 3600 * 1000 / self.beat_rate  # ms
        self.nominal_tick_interval = self.expected_interval / 2  # ms
        
        # Analysis buffers and state with increased capacity
        self.tick_times = deque(maxlen=5000)  # Store more ticks for better statistics
        self.tick_intervals = deque(maxlen=5000)
        self.amplitudes = deque(maxlen=5000)
        self.beat_errors = deque(maxlen=500)
        
        # Advanced detection parameters
        self.min_consecutive_ticks = 8  # Reduced to detect patterns earlier
        self.noise_floor = 0.03  # Lower threshold for better sensitivity
        self.signal_quality = 0
        
        # Confidence metrics
        self.interval_consistency = 0
        self.amplitude_consistency = 0
        self.overall_confidence = 0
        
        # Averaging and filtering parameters
        self.avg_window_size = 30  # Increased for better stability
        self.outlier_threshold = 3.0  # More permissive for weak signals
        
        # FFT analysis parameters
        self.fft_buffer_size = 10  # Increased buffer for better frequency resolution
        self.fft_sample_rate = 44100
        self.fft_buffer = None
        
        # Audio buffer for analysis
        self.buffer_size = 10  # Increased from 8 seconds
        self.audio_buffer = None
        self.buffer_index = 0
        
        # Timing tracking
        self.start_time = time.time()
        self.last_tick_time = None
        
        # Storage for detection data
        self.raw_tick_data = []
        self.filtered_tick_data = []
        
        # Analysis results
        self.results = {
            'rate': 0,
            'beat_error': 0,
            'amplitude': 0,
            'confidence': 0,
            'tick_times': [],
            'tick_intervals': [],
            'tick_patterns': [],
            'quality_metrics': {
                'interval_consistency': 0,
                'amplitude_consistency': 0,
                'signal_to_noise': 0
            }
        }
        
        # Analysis state
        self.has_analysis_data = False
        
    def set_beat_rate(self, beat_rate):
        """Set the watch beat rate in BPH"""
        self.beat_rate = beat_rate
        self.expected_interval = 3600 * 1000 / self.beat_rate  # ms
        self.nominal_tick_interval = self.expected_interval / 2  # ms
        
    def process_audio_data(self, audio_data, sample_rate):
        """Process incoming audio data with ultra-sensitive detection"""
        # Initialize buffers if needed
        if self.fft_buffer is None:
            self.fft_buffer = np.zeros(self.fft_buffer_size * sample_rate)
            self.fft_sample_rate = sample_rate
        
        if self.audio_buffer is None:
            self.audio_buffer = np.zeros(self.buffer_size * sample_rate)
        
        # Update FFT buffer
        data_len = len(audio_data)
        if data_len < len(self.fft_buffer):
            self.fft_buffer = np.roll(self.fft_buffer, -data_len)
            self.fft_buffer[-data_len:] = audio_data
        else:
            self.fft_buffer = audio_data[-len(self.fft_buffer):]
        
        # Update audio buffer
        if data_len >= len(self.audio_buffer):
            self.audio_buffer = audio_data[-len(self.audio_buffer):]
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -data_len)
            self.audio_buffer[-data_len:] = audio_data
        
        # Auto-detect beat rate periodically
        self.auto_detect_beat_rate(sample_rate)
        
        # Detect ticks with ultra-sensitive algorithm
        tick_times, tick_quality = self.detect_ticks(audio_data, sample_rate)
        
        # Process detected ticks
        current_time = time.time() - self.start_time
        for i, tick_time in enumerate(tick_times):
            # Convert tick time to absolute time
            absolute_tick_time = current_time - (len(audio_data) / sample_rate) + tick_time
            
            # Store tick data
            self.tick_times.append(absolute_tick_time)
            
            # Calculate interval if we have previous ticks
            if len(self.tick_times) > 1:
                interval = (self.tick_times[-1] - self.tick_times[-2]) * 1000  # Convert to ms
                self.tick_intervals.append(interval)
            
            # Store quality as amplitude proxy
            if i < len(tick_quality):
                self.amplitudes.append(tick_quality[i] * 100)  # Scale to 0-100
        
        # Analyze timing if we have enough data
        if len(self.tick_times) > self.min_consecutive_ticks:
            self.analyze_timing()
            self.has_analysis_data = True
            self.calculate_confidence()
    
    def detect_ticks(self, audio_data, sample_rate):
        """Ultra-sensitive tick detection optimized for very faint watch sounds"""
        # Step 1: Adaptive Pre-amplification
        signal_level = np.percentile(np.abs(audio_data), 95)
        if signal_level < 0.02:  # Extremely weak signal
            pre_amp_factor = min(20.0, 0.5 / max(0.001, signal_level))
            audio_data = audio_data * pre_amp_factor
        elif signal_level < 0.05:  # Very weak signal
            pre_amp_factor = min(12.0, 0.3 / max(0.001, signal_level))
            audio_data = audio_data * pre_amp_factor
        elif signal_level < 0.1:  # Weak signal
            pre_amp_factor = min(6.0, 0.2 / max(0.001, signal_level))
            audio_data = audio_data * pre_amp_factor
        
        # Step 2: Multi-band Analysis - Target specific frequency bands where watch ticks occur
        # Create multiple bandpass filters for different tick frequencies
        sos_1k5 = signal.butter(6, [1000, 2000], 'bandpass', fs=sample_rate, output='sos')
        sos_3k = signal.butter(6, [2500, 3500], 'bandpass', fs=sample_rate, output='sos')
        sos_6k = signal.butter(6, [5000, 7000], 'bandpass', fs=sample_rate, output='sos')
        
        # Apply filters to isolate frequency bands
        band_1k5 = signal.sosfilt(sos_1k5, audio_data)
        band_3k = signal.sosfilt(sos_3k, audio_data)
        band_6k = signal.sosfilt(sos_6k, audio_data)
        
        # Combine with weight emphasis on mid-frequencies (most watch ticks)
        filtered_signal = band_1k5 * 1.2 + band_3k * 1.5 + band_6k * 0.8
        
        # Step 3: Apply phase-preserving filtering to sharpen transients
        sos_high = signal.butter(4, 800, 'highpass', fs=sample_rate, output='sos')
        filtered_signal = signal.sosfiltfilt(sos_high, filtered_signal)
        
        # Step 4: Apply wavelet denoising if available
        filtered_signal = self._wavelet_denoise(filtered_signal)
        
        # Step 5: Apply pre-emphasis to enhance transients
        pre_emphasis = 0.98
        emphasized = np.append(filtered_signal[0], 
                              filtered_signal[1:] - pre_emphasis * filtered_signal[:-1])
        
        # Step 6: Calculate envelope and frequency information
        analytic_signal = signal.hilbert(emphasized)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Calculate instantaneous frequency information
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
        # Pad to maintain length
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), 'edge')
        
        # Calculate frequency changes (transients indicate ticks)
        frequency_diff = np.abs(np.diff(instantaneous_frequency))
        frequency_diff = np.pad(frequency_diff, (0, 1), 'edge')
        
        # Normalize for easier thresholding
        if np.max(frequency_diff) > 0:
            frequency_diff = frequency_diff / np.max(frequency_diff)
        
        # Normalize amplitude envelope
        if np.max(amplitude_envelope) > 0:
            norm_amplitude = amplitude_envelope / np.max(amplitude_envelope)
        else:
            norm_amplitude = amplitude_envelope
        
        # Step 7: Create composite detection signal (frequency transients + amplitude)
        composite_signal = norm_amplitude * 0.3 + frequency_diff * 0.7
        
        # Step 8: Apply short-window smoothing to preserve transients
        win_length = max(5, int(sample_rate * 0.0005))  # 0.5ms window
        if win_length % 2 == 0:
            win_length += 1  # Ensure odd window length
        
        try:
            smoothed_signal = signal.savgol_filter(composite_signal, win_length, 2)
        except ValueError:
            smoothed_signal = composite_signal
        
        # Step 9: Multi-threshold peak detection
        signal_mean = np.mean(smoothed_signal)
        signal_std = np.std(smoothed_signal)
        signal_max = np.max(smoothed_signal)
        
        # Two thresholds: strong peaks (primary) and weaker candidates (secondary)
        primary_threshold = max(signal_mean + 2.0 * signal_std, signal_max * 0.4)
        secondary_threshold = max(signal_mean + 1.0 * signal_std, signal_max * 0.2)
        
        # Find primary peaks - high confidence ticks
        primary_peaks, _ = signal.find_peaks(
            smoothed_signal,
            height=primary_threshold,
            distance=int(sample_rate * 0.08)  # ~80ms minimum between ticks
        )
        
        # Find all potential peaks - including weaker candidates
        all_peaks, _ = signal.find_peaks(
            smoothed_signal,
            height=secondary_threshold,
            distance=int(sample_rate * 0.08)
        )
        
        # Step 10: Pattern-based validation
        # If we have enough primary peaks, use them to estimate the tick interval
        if len(primary_peaks) >= 3:
            # Calculate intervals between primary peaks
            intervals = np.diff(primary_peaks) / sample_rate
            median_interval = np.median(intervals)
            
            # Check if interval matches standard watch rates
            standard_intervals = [0.1, 0.125, 0.143, 0.167, 0.2]  # From 36000 to 18000 BPH
            closest_standard = min(standard_intervals, 
                                  key=lambda x: abs(x - median_interval))
            interval_error = abs(median_interval - closest_standard) / closest_standard
            
            # If we match a standard rate, use pattern matching to find missing ticks
            if interval_error < 0.2:  # Within 20% of standard rate
                expected_interval = closest_standard
                expected_samples = int(expected_interval * sample_rate)
                
                # Extract peaks based on pattern
                validated_peaks = list(primary_peaks)
                
                # Look forward and backward from primary peaks
                for peak in primary_peaks:
                    # Look forward
                    for i in range(1, 5):  # Up to 5 ticks ahead
                        expected_pos = peak + i * expected_samples
                        if expected_pos >= len(smoothed_signal):
                            continue
                        
                        # Check window around expected position
                        window_size = int(expected_samples * 0.2)  # 20% tolerance
                        start = max(0, expected_pos - window_size)
                        end = min(len(smoothed_signal), expected_pos + window_size)
                        
                        if start >= end:
                            continue
                        
                        # Find highest peak in window
                        window_max = np.argmax(smoothed_signal[start:end]) + start
                        
                        # Add if above threshold and not already found
                        if (smoothed_signal[window_max] > secondary_threshold and 
                            window_max not in validated_peaks):
                            validated_peaks.append(window_max)
                    
                    # Look backward
                    for i in range(1, 5):  # Up to 5 ticks behind
                        expected_pos = peak - i * expected_samples
                        if expected_pos < 0:
                            continue
                        
                        window_size = int(expected_samples * 0.2)
                        start = max(0, expected_pos - window_size)
                        end = min(len(smoothed_signal), expected_pos + window_size)
                        
                        if start >= end:
                            continue
                        
                        window_max = np.argmax(smoothed_signal[start:end]) + start
                        
                        if (smoothed_signal[window_max] > secondary_threshold and 
                            window_max not in validated_peaks):
                            validated_peaks.append(window_max)
                
                # Use these pattern-matched peaks
                peaks = np.array(sorted(validated_peaks))
            else:
                # If no pattern match, use all detected peaks
                peaks = all_peaks
        else:
            # Not enough primary peaks, use all candidates
            peaks = all_peaks
        
        # Step 11: Calculate quality metrics for each peak
        if len(peaks) > 0:
            # Get heights from smoothed signal
            peak_heights = smoothed_signal[peaks]
            
            # Normalize heights
            if np.max(peak_heights) > 0:
                normalized_heights = peak_heights / np.max(peak_heights)
            else:
                normalized_heights = peak_heights
            
            # Calculate interval consistency if possible
            if len(peaks) > 1:
                intervals = np.diff(peaks) / sample_rate
                if np.mean(intervals) > 0:
                    interval_consistency = 1.0 - min(1.0, np.std(intervals) / np.mean(intervals))
                else:
                    interval_consistency = 0.0
            else:
                interval_consistency = 0.0
            
            # Combined quality score
            qualities = 0.7 * normalized_heights
            if len(peaks) > 1:
                # Add consistency component to all qualities
                qualities = qualities + (0.3 * interval_consistency)
            
            # Bonus for primary (high confidence) peaks
            for i, peak in enumerate(peaks):
                if peak in primary_peaks:
                    qualities[i] = min(1.0, qualities[i] * 1.3)
        else:
            qualities = np.array([])
        
        # Return detected ticks and their quality
        tick_times = peaks / sample_rate
        tick_quality = qualities
        
        return tick_times, tick_quality
    
    def _wavelet_denoise(self, data, wavelet='db4', level=3):
        """Apply wavelet denoising to enhance signal quality"""
        if pywt is None:
            return data
        
        try:
            # Decompose signal using wavelet transform
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
            # Apply thresholding to detail coefficients (leave approximation untouched)
            for i in range(1, len(coeffs)):
                # Calculate adaptive threshold based on coefficient statistics
                abs_coeffs = np.abs(coeffs[i])
                if len(abs_coeffs) > 0:
                    threshold = np.median(abs_coeffs) / 0.6745 * np.sqrt(2 * np.log(len(data)))
                    coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # Reconstruct denoised signal
            return pywt.waverec(coeffs, wavelet)
        except Exception as e:
            print(f"Wavelet denoising error: {e}")
            return data
    
    def auto_detect_beat_rate(self, sample_rate):
        """Automatically detect watch beat rate using FFT analysis"""
        # Only run periodically to avoid constant recalculation
        if hasattr(self, 'last_fft_time') and time.time() - self.last_fft_time < 2.0:
            return
            
        self.last_fft_time = time.time()
        
        # Apply filtering to isolate tick sounds
        sos_high = signal.butter(4, 800, 'highpass', fs=sample_rate, output='sos')
        high_passed = signal.sosfilt(sos_high, self.fft_buffer)
        
        sos_band = signal.butter(6, [1500, 8000], 'bandpass', fs=sample_rate, output='sos')
        filtered_data = signal.sosfilt(sos_band, high_passed)
        
        # Get signal envelope
        analytic_signal = signal.hilbert(filtered_data)
        envelope = np.abs(analytic_signal)
        
        # Downsample for faster FFT processing
        downsample_factor = 4
        downsampled = envelope[::downsample_factor]
        ds_rate = sample_rate / downsample_factor
        
        # Calculate FFT
        n = len(downsampled)
        freqs = np.fft.rfftfreq(n, 1/ds_rate)
        fft_values = np.abs(np.fft.rfft(downsampled))
        
        # Look for peaks in frequency range of watch tick rates
        # Common rates: 18000, 21600, 25200, 28800, 36000 BPH (5-10 Hz)
        min_freq = 4.0  # Hz
        max_freq = 12.0  # Hz
        
        # Filter to expected frequency range
        valid_range = (freqs >= min_freq) & (freqs <= max_freq)
        if not any(valid_range):
            return
            
        valid_freqs = freqs[valid_range]
        valid_ffts = fft_values[valid_range]
        
        # Find strongest frequency components
        if len(valid_ffts) > 0:
            peak_indices = signal.find_peaks(valid_ffts, height=np.max(valid_ffts)*0.5)[0]
            
            if len(peak_indices) > 0:
                # Get strongest peak
                strongest_idx = peak_indices[np.argmax(valid_ffts[peak_indices])]
                peak_freq = valid_freqs[strongest_idx]
                
                # Convert to BPH
                detected_bph = peak_freq * 3600 / 2
                
                # Compare to standard rates
                standard_rates = [18000, 21600, 25200, 28800, 36000]
                closest_rate = min(standard_rates, key=lambda x: abs(x - detected_bph))
                
                # Update if detection is confident
                error_percent = abs(detected_bph - closest_rate) / closest_rate
                if error_percent < 0.1:  # Within 10% of standard rate
                    if self.beat_rate != closest_rate:
                        self.set_beat_rate(closest_rate)
                        print(f"Auto-detected beat rate: {closest_rate} BPH")
    
    def calculate_confidence(self):
        """Calculate confidence metrics for analysis results"""
        # Minimum data check
        if len(self.tick_intervals) < 5:
            self.overall_confidence = 0
            return
        
        # Recent intervals for analysis
        recent_intervals = list(self.tick_intervals)[-min(len(self.tick_intervals), 50):]
        recent_amplitudes = list(self.amplitudes)[-min(len(self.amplitudes), 50):]
        
        # Interval consistency (lower coefficient of variation = higher consistency)
        if len(recent_intervals) > 2:
            mean_interval = np.mean(recent_intervals)
            if mean_interval > 0:
                std_interval = np.std(recent_intervals)
                cv_interval = std_interval / mean_interval
                self.interval_consistency = max(0, min(10, 10 * (1 - cv_interval)))
            else:
                self.interval_consistency = 0
        else:
            self.interval_consistency = 0
        
        # Amplitude consistency
        if len(recent_amplitudes) > 2:
            mean_amplitude = np.mean(recent_amplitudes)
            if mean_amplitude > 0:
                std_amplitude = np.std(recent_amplitudes)
                cv_amplitude = std_amplitude / mean_amplitude
                self.amplitude_consistency = max(0, min(10, 10 * (1 - cv_amplitude)))
            else:
                self.amplitude_consistency = 0
        else:
            self.amplitude_consistency = 0
        
        # Signal-to-noise proxy based on amplitude
        snr_score = min(10, max(0, np.mean(recent_amplitudes) / 5))
        
        # Combined confidence metric
        # Weights: 50% interval consistency, 30% amplitude consistency, 20% SNR
        self.overall_confidence = (0.5 * self.interval_consistency + 
                                  0.3 * self.amplitude_consistency + 
                                  0.2 * snr_score)
        
        # Update confidence in results
        self.results['confidence'] = self.overall_confidence
        self.results['quality_metrics']['interval_consistency'] = self.interval_consistency
        self.results['quality_metrics']['amplitude_consistency'] = self.amplitude_consistency
        self.results['quality_metrics']['signal_to_noise'] = snr_score
    
    def reset(self):
        """Reset all analysis data"""
        # Clear data structures
        self.tick_times.clear()
        self.tick_intervals.clear()
        self.amplitudes.clear()
        self.beat_errors.clear()
        
        # Reset metrics
        self.interval_consistency = 0
        self.amplitude_consistency = 0
        self.overall_confidence = 0
        
        # Reset results
        self.results = {
            'rate': 0,
            'beat_error': 0,
            'amplitude': 0,
            'confidence': 0,
            'tick_times': [],
            'tick_intervals': [],
            'tick_patterns': [],
            'quality_metrics': {
                'interval_consistency': 0,
                'amplitude_consistency': 0,
                'signal_to_noise': 0
            }
        }
        
        # Reset state
        self.has_analysis_data = False
        self.start_time = time.time()
        self.last_tick_time = None
        
        print("Analysis data reset")
    
    def analyze_timing(self):
        """Calculate timegrapher metrics with better handling of weak signals"""
        if len(self.tick_intervals) < 5:
            return
            
        # Get recent intervals and amplitudes for analysis
        recent_intervals = list(self.tick_intervals)[-min(len(self.tick_intervals), 50):]
        recent_amplitudes = list(self.amplitudes)[-min(len(self.amplitudes), 50):]
        
        # Remove outliers (more permissive with weak signals)
        if len(recent_intervals) > 10:
            mean_interval = np.mean(recent_intervals)
            std_interval = np.std(recent_intervals)
            
            if std_interval > 0:
                # Keep intervals within threshold of mean
                filtered_intervals = []
                filtered_amplitudes = []
                
                for i, interval in enumerate(recent_intervals):
                    if i < len(recent_amplitudes):
                        # Adjust threshold based on signal quality
                        if self.overall_confidence < 3:
                            # Very weak signal - be more permissive
                            threshold = self.outlier_threshold * 1.5
                        elif self.overall_confidence < 6:
                            # Medium quality signal
                            threshold = self.outlier_threshold * 1.2
                        else:
                            # Good signal - use standard threshold
                            threshold = self.outlier_threshold
                            
                        if abs(interval - mean_interval) <= threshold * std_interval:
                            filtered_intervals.append(interval)
                            if i < len(recent_amplitudes):
                                filtered_amplitudes.append(recent_amplitudes[i])
                
                if len(filtered_intervals) > 3:
                    recent_intervals = filtered_intervals
                    recent_amplitudes = filtered_amplitudes
        
        # Calculate rate
        if len(recent_intervals) > 0:
            # Average interval in seconds
            avg_interval_ms = np.mean(recent_intervals)
            
            # Convert to BPH (Beats Per Hour)
            # Each complete tick-tock cycle is one beat
            # So interval between consecutive ticks is half a beat period
            if avg_interval_ms > 0:
                beats_per_second = 1000 / (avg_interval_ms * 2)
                beats_per_hour = beats_per_second * 3600
                
                # Look for nearest standard rate
                standard_rates = [18000, 19800, 21600, 25200, 28800, 36000]
                closest_rate = min(standard_rates, key=lambda x: abs(x - beats_per_hour))
                
                # Calculate seconds per day deviation
                rate_difference = beats_per_hour - closest_rate
                seconds_per_day = (rate_difference / closest_rate) * 86400
                
                # Update results
                self.results['rate'] = beats_per_hour
                self.results['beat_error'] = self.calculate_beat_error(recent_intervals)
                
                # Estimate amplitude based on quality metrics (simple approximation)
                # Scale to typical amplitude range (degrees of swing)
                if len(recent_amplitudes) > 0:
                    normalized_amplitude = np.mean(recent_amplitudes) / 100  # Scale back from 0-100
                    amplitude_estimate = 200 + (normalized_amplitude * 115)  # Map to 200-315°
                    self.results['amplitude'] = amplitude_estimate
                
                # Update result collections
                self.results['tick_times'] = list(self.tick_times)
                self.results['tick_intervals'] = list(self.tick_intervals)
    
    def calculate_beat_error(self, intervals):
        """Calculate beat error from intervals"""
        if len(intervals) < 4:
            return 0
            
        # Group intervals into pairs to detect asymmetry
        beat_errors = []
        for i in range(0, len(intervals) - 1, 2):
            if i + 1 < len(intervals):
                # Each pair of consecutive ticks forms tick-tock
                tick_interval = intervals[i]
                tock_interval = intervals[i + 1]
                total_interval = tick_interval + tock_interval
                
                if total_interval > 0:
                    # Calculate asymmetry - perfect would be 0.5 (equal intervals)
                    ratio = tick_interval / total_interval
                    asymmetry = abs(ratio - 0.5) * 2  # Scale to 0-1
                    beat_errors.append(asymmetry * 100)  # Convert to percentage
        
        # Return average beat error
        if beat_errors:
            return np.mean(beat_errors)
        return 0
    
    def get_results(self):
        """Get current analysis results"""
        return self.results
        
    def has_data(self):
        """Check if we have analysis data"""
        return self.has_analysis_data
        
    def export_results(self, filename):
        """Export analysis results to CSV"""
        if not self.has_analysis_data:
            print("No data to export")
            return False
            
        try:
            # Create DataFrame from our measurements
            data = {
                'Time': list(self.tick_times),
                'Interval': list(self.tick_intervals),
                'Amplitude': list(self.amplitudes),
            }
            
            # Add beat error if available
            if len(self.beat_errors) > 0:
                data['Beat Error'] = list(self.beat_errors)
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add metadata
            metadata = pd.DataFrame({
                'Parameter': ['Beat Rate', 'Seconds per day', 'Average Amplitude', 'Confidence'],
                'Value': [
                    f"{self.results['rate']:.0f} BPH",
                    f"{self.results.get('seconds_per_day', 0):.1f} s/d",
                    f"{self.results['amplitude']:.1f}°",
                    f"{self.results['confidence']:.1f}/10"
                ]
            })
            
            # Export to CSV
            with open(filename, 'w') as f:
                # Write metadata
                f.write("# Timegrapher Analysis Results\n")
                metadata.to_csv(f, index=False)
                f.write("\n# Measurement Data\n")
                df.to_csv(f, index=False)
                
            print(f"Results exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False
