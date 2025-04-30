#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Manager for the Desktop Timegrapher Application
Handles audio input device selection and streaming
"""

import numpy as np
import sounddevice as sd
from threading import Lock

class AudioManager:
    """Manages audio input for timegrapher analysis with ultra-sensitive detection"""
    
    def __init__(self):
        """Initialize the audio manager with enhanced capabilities"""
        self.stream = None
        self.callback_function = None
        self.device_index = None
        self.lock = Lock()
        
        # Enhanced audio processing settings - MAXIMUM SENSITIVITY
        self.gain_level = 30.0  # Extreme amplification for very weak signals
        self.use_noise_reduction = True
        self.noise_profile = None  # Will store background noise profile
        self.noise_floor = None    # Will track noise floor level
        self.calibration_mode = True  # Start with calibration mode for better initial results
        
        # Audio buffer for noise cancellation
        self.buffer_size = 8  # increased to 8 seconds for better noise profile
        self.sample_rate = 44100
        self.audio_buffer = None
        
        # Signal enhancement parameters - optimized for watch ticks
        self.dynamic_range_compression = True
        self.compressor_threshold = 0.02  # Lower threshold to catch weaker signals
        self.compressor_ratio = 8.0       # Stronger compression for better detection
        
    def get_input_devices(self):
        """Get a list of available input (microphone) devices"""
        devices = []
        device_list = sd.query_devices()
        
        for i, device in enumerate(device_list):
            # Only include input devices
            if device['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
                
        return devices
    
    def set_input_device(self, device_index):
        """Set the input device to use"""
        with self.lock:
            was_running = self.stream is not None and self.stream.active
            
            # Stop stream if running
            if was_running:
                self.stop_stream()
            
            self.device_index = device_index
            
            # Restart stream if it was running
            if was_running and self.callback_function is not None:
                self.start_stream(
                    self.stream.samplerate,
                    self.stream.blocksize,
                    self.callback_function
                )
                
        return True
    
    def start_stream(self, sample_rate, frame_size, callback):
        """Start the audio input stream with overflow handling and ultra-sensitive settings"""
        with self.lock:
            if self.stream is not None and self.stream.active:
                self.stop_stream()
            
            self.callback_function = callback
            self.overflow_count = 0
            self.last_warning_time = 0
            self.sample_rate = sample_rate
            
            # Initialize calibration if needed
            if self.noise_profile is None:
                self.calibration_mode = True
                print("Initializing noise calibration...")
            
            # Create audio stream with reduced default sensitivity
            try:
                # First, try to get device info to set appropriate parameters
                device_info = None
                if self.device_index is not None:
                    try:
                        device_info = sd.query_devices(self.device_index)
                    except:
                        pass
                
                # If we have device info, use it to determine defaults
                if device_info is not None and 'default_high_input_latency' in device_info:
                    latency = device_info['default_high_input_latency']
                else:
                    latency = 0.1  # Default to a higher latency to avoid overflow
                
                self.stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=sample_rate,
                    blocksize=frame_size,
                    channels=1,  # Mono input is sufficient
                    dtype='float32',
                    latency=latency,  # Use higher latency to prevent overflow
                    callback=self._audio_callback
                )
                
                self.stream.start()
                
            except Exception as e:
                print(f"Error starting audio stream: {str(e)}")
                # Try with default parameters if custom ones fail
                self.stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=sample_rate,
                    blocksize=frame_size,
                    channels=1,
                    dtype='float32',
                    callback=self._audio_callback
                )
                
                self.stream.start()
            
        return True
    
    def stop_stream(self):
        """Stop the audio input stream"""
        with self.lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
        return True
    
    def _audio_callback(self, indata, frames, time, status):
        """Internal callback for audio data processing with ultra-sensitive enhancement"""
        import time as pytime  # Import time module in function to avoid conflict
        import scipy.signal as signal
        
        # Handle overflow errors
        if status:
            if status.input_overflow:
                self.overflow_count += 1
                current_time = pytime.time()
                
                # Only display a warning once every 2 seconds to avoid flood
                if current_time - self.last_warning_time > 2:
                    print(f"Input overflow detected ({self.overflow_count} times). Try reducing microphone sensitivity.")
                    self.last_warning_time = current_time
            else:
                print(f"Status: {status}")
        
        if self.callback_function is not None:
            try:
                # Convert to floating point and normalize
                audio_data = np.squeeze(indata).astype(np.float64)
                
                # Initialize or update audio buffer for noise profiling
                if self.audio_buffer is None:
                    self.audio_buffer = np.zeros((self.sample_rate * self.buffer_size,))
                
                # Update buffer (shift and add new data)
                if len(audio_data) < len(self.audio_buffer):
                    self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                    self.audio_buffer[-len(audio_data):] = audio_data
                else:
                    # If more data than buffer, just use the latest buffer-sized chunk
                    self.audio_buffer = audio_data[-len(self.audio_buffer):]
                
                # If in calibration mode, update noise profile
                if self.calibration_mode and self.noise_profile is None:
                    # Use the first 1 second of audio as noise profile if signal is quiet
                    if np.max(np.abs(audio_data)) < 0.1:  # Low amplitude = probably just noise
                        self.noise_profile = audio_data.copy()
                        self.noise_floor = np.mean(np.abs(self.noise_profile))
                        print("Noise profile calibrated")
                        self.calibration_mode = False
                
                # Apply noise reduction if enabled and we have a noise profile
                if self.use_noise_reduction and self.noise_profile is not None:
                    # Simple spectral subtraction
                    n_fft = min(2048, len(audio_data))
                    
                    # Get audio spectrum
                    audio_spec = np.fft.rfft(audio_data, n=n_fft)
                    audio_pow = np.abs(audio_spec) ** 2
                    
                    # Get noise spectrum (use same length)
                    noise_spec = np.fft.rfft(self.noise_profile[:n_fft] if len(self.noise_profile) >= n_fft 
                                             else np.pad(self.noise_profile, (0, n_fft - len(self.noise_profile))), n=n_fft)
                    noise_pow = np.abs(noise_spec) ** 2
                    
                    # Compute suppression factor (Wiener filter approximation)
                    snr = np.maximum(audio_pow / (noise_pow + 1e-10) - 1, 0)  # Avoid negative SNR
                    gain = snr / (snr + 1)  # Wiener gain
                    
                    # Apply suppression with spectral floor
                    enhanced_spec = audio_spec * np.maximum(gain, 0.1)  # Minimum gain to avoid musical noise
                    
                    # Inverse FFT and take real part
                    audio_data = np.real(np.fft.irfft(enhanced_spec, len(audio_data)))
                
                # Apply multi-band dynamic range compression if enabled
                if self.dynamic_range_compression:
                    # Split audio into frequency bands
                    low_band = signal.butter(4, 800, 'lowpass', fs=self.sample_rate, output='sos')
                    mid_band = signal.butter(4, [800, 3000], 'bandpass', fs=self.sample_rate, output='sos')
                    high_band = signal.butter(4, 3000, 'highpass', fs=self.sample_rate, output='sos')
                    
                    low_audio = signal.sosfilt(low_band, audio_data)
                    mid_audio = signal.sosfilt(mid_band, audio_data)
                    high_audio = signal.sosfilt(high_band, audio_data)
                    
                    # Apply different compression to each band (strongest on mid where ticks are)
                    low_audio = self._apply_compression(low_audio, self.compressor_threshold * 1.5, self.compressor_ratio * 0.7)
                    mid_audio = self._apply_compression(mid_audio, self.compressor_threshold * 0.7, self.compressor_ratio * 1.5)
                    high_audio = self._apply_compression(high_audio, self.compressor_threshold, self.compressor_ratio)
                    
                    # Recombine with different weights (emphasize mid-range where ticks usually are)
                    audio_data = low_audio * 0.6 + mid_audio * 1.5 + high_audio * 0.8
                
                # Apply adaptive gain based on current volume with soft clipping
                current_level = np.percentile(np.abs(audio_data), 95)
                if current_level < 0.01:  # Very quiet
                    # Apply stronger gain for very quiet signals
                    adaptive_gain = min(20.0, self.gain_level * 2.0)
                elif current_level < 0.05:  # Quiet
                    # Apply normal gain
                    adaptive_gain = self.gain_level
                else:  # Loud enough
                    # Apply reduced gain
                    adaptive_gain = max(1.0, self.gain_level * 0.5)
                
                # Apply gain with soft clipping to prevent distortion
                audio_data = audio_data * adaptive_gain
                audio_data = np.tanh(audio_data)  # Soft clipping with hyperbolic tangent
                
                # Pass the enhanced audio to the callback
                self.callback_function(audio_data)
                
            except Exception as e:
                print(f"Error in audio callback: {str(e)}")
    
    def _apply_compression(self, audio, threshold, ratio):
        """Apply dynamic range compression to audio signal"""
        # Calculate signal magnitude
        magnitude = np.abs(audio)
        
        # Calculate gain reduction
        mask = magnitude > threshold
        gain = np.ones_like(audio)
        gain[mask] = threshold + (magnitude[mask] - threshold) / ratio
        gain[mask] = gain[mask] / magnitude[mask]
        
        # Apply gain reduction
        return audio * gain
            
    def set_gain(self, gain_level):
        """Set the gain level for input amplification"""
        with self.lock:
            self.gain_level = max(1.0, min(20.0, gain_level))  # Limit to reasonable range
            return self.gain_level
    
    def toggle_noise_reduction(self, enabled=None):
        """Toggle noise reduction on/off"""
        with self.lock:
            if enabled is not None:
                self.use_noise_reduction = enabled
            else:
                self.use_noise_reduction = not self.use_noise_reduction
            return self.use_noise_reduction
    
    def calibrate_noise(self):
        """Start noise profile calibration"""
        with self.lock:
            self.calibration_mode = True
            self.noise_profile = None
            print("Starting noise calibration - please ensure environment is quiet")
            return True
    
    def reset_audio_processing(self):
        """Reset all audio processing parameters to defaults"""
        with self.lock:
            self.gain_level = 5.0
            self.use_noise_reduction = True
            self.noise_profile = None
            self.calibration_mode = False
            self.dynamic_range_compression = True
            self.compressor_threshold = 0.05
            self.compressor_ratio = 4.0
            print("Audio processing parameters reset to defaults")
            return True
    
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.stop_stream()
