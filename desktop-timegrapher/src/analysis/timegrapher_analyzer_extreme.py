#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de Timegrapher con detector extremo de ticks
Versión especial optimizada para señales extremadamente débiles
"""

import numpy as np
import pandas as pd
from scipy import signal
from collections import deque
import time
from .extreme_tick_detector import ExtremeTickDetector

class TimegrapherAnalyzerExtreme:
    """Analizador de timegrapher mejorado para señales extremadamente débiles"""
    
    def __init__(self):
        """Inicializar analizador"""
        # Parámetros por defecto
        self.beat_rate = 28800  # BPH por defecto
        self.expected_interval = 3600 * 1000 / self.beat_rate  # ms
        self.nominal_tick_interval = self.expected_interval / 2  # ms
        
        # Detector extremo de ticks
        self.tick_detector = ExtremeTickDetector(self.beat_rate)
        
        # Buffers de análisis
        self.tick_times = deque(maxlen=5000)
        self.tick_intervals = deque(maxlen=5000)
        self.amplitudes = deque(maxlen=5000)
        self.beat_errors = deque(maxlen=500)
        
        # Métricas
        self.interval_consistency = 0
        self.amplitude_consistency = 0
        self.overall_confidence = 0
        
        # Parámetros de análisis
        self.min_consecutive_ticks = 8
        self.avg_window_size = 30
        self.outlier_threshold = 3.0
        
        # FFT para análisis de frecuencia
        self.fft_buffer_size = 10
        self.fft_sample_rate = 44100
        self.fft_buffer = None
        
        # Audio buffer
        self.buffer_size = 10
        self.audio_buffer = None
        
        # Estado
        self.start_time = time.time()
        self.has_analysis_data = False
        
        # Resultados
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
    
    def set_beat_rate(self, beat_rate):
        """Establecer tasa de beats en BPH"""
        self.beat_rate = beat_rate
        self.expected_interval = 3600 * 1000 / self.beat_rate  # ms
        self.nominal_tick_interval = self.expected_interval / 2  # ms
        self.tick_detector.set_expected_bph(beat_rate)
    
    def process_audio_data(self, audio_data, sample_rate):
        """Procesar datos de audio con detector extremo"""
        # Inicializar buffers si es necesario
        if self.fft_buffer is None:
            self.fft_buffer = np.zeros(self.fft_buffer_size * sample_rate)
            self.fft_sample_rate = sample_rate
        
        if self.audio_buffer is None:
            self.audio_buffer = np.zeros(self.buffer_size * sample_rate)
        
        # Actualizar FFT buffer
        data_len = len(audio_data)
        if data_len < len(self.fft_buffer):
            self.fft_buffer = np.roll(self.fft_buffer, -data_len)
            self.fft_buffer[-data_len:] = audio_data
        else:
            self.fft_buffer = audio_data[-len(self.fft_buffer):]
        
        # Actualizar audio buffer
        if data_len >= len(self.audio_buffer):
            self.audio_buffer = audio_data[-len(self.audio_buffer):]
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -data_len)
            self.audio_buffer[-data_len:] = audio_data
        
        # Detección extrema de ticks
        tick_times, tick_quality = self.tick_detector.process_audio(audio_data, sample_rate)
        
        # Si tenemos suficientes datos, realizar análisis
        detector_results = self.tick_detector.get_results()
        self.tick_times = deque(detector_results['tick_times'], maxlen=5000)
        self.tick_intervals = deque(detector_results['tick_intervals'], maxlen=5000)
        self.amplitudes = deque(detector_results['tick_amplitudes'], maxlen=5000)
        
        if len(self.tick_times) > self.min_consecutive_ticks:
            self.analyze_timing()
            self.has_analysis_data = True
            self.calculate_confidence()
    
    def auto_detect_beat_rate(self, sample_rate):
        """Auto-detectar tasa de beats usando FFT"""
        # Solo ejecutar periódicamente
        if hasattr(self, 'last_fft_time') and time.time() - self.last_fft_time < 2.0:
            return
            
        self.last_fft_time = time.time()
        
        # Filtrar para aislar sonidos de ticks
        sos_high = signal.butter(4, 800, 'highpass', fs=sample_rate, output='sos')
        high_passed = signal.sosfilt(sos_high, self.fft_buffer)
        
        sos_band = signal.butter(6, [1500, 8000], 'bandpass', fs=sample_rate, output='sos')
        filtered_data = signal.sosfilt(sos_band, high_passed)
        
        # Obtener envolvente
        analytic_signal = signal.hilbert(filtered_data)
        envelope = np.abs(analytic_signal)
        
        # Submuestreo para FFT más rápida
        downsample_factor = 4
        downsampled = envelope[::downsample_factor]
        ds_rate = sample_rate / downsample_factor
        
        # Calcular FFT
        n = len(downsampled)
        freqs = np.fft.rfftfreq(n, 1/ds_rate)
        fft_values = np.abs(np.fft.rfft(downsampled))
        
        # Buscar picos en rango de frecuencias de relojes
        min_freq = 4.0  # Hz
        max_freq = 12.0  # Hz
        
        # Filtrar rango esperado
        valid_range = (freqs >= min_freq) & (freqs <= max_freq)
        if not any(valid_range):
            return
            
        valid_freqs = freqs[valid_range]
        valid_ffts = fft_values[valid_range]
        
        # Encontrar componentes de frecuencia fuertes
        if len(valid_ffts) > 0:
            peak_indices = signal.find_peaks(valid_ffts, height=np.max(valid_ffts)*0.5)[0]
            
            if len(peak_indices) > 0:
                # Obtener pico más fuerte
                strongest_idx = peak_indices[np.argmax(valid_ffts[peak_indices])]
                peak_freq = valid_freqs[strongest_idx]
                
                # Convertir a BPH
                detected_bph = peak_freq * 3600 / 2
                
                # Comparar con tasas estándar
                standard_rates = [18000, 21600, 25200, 28800, 36000]
                closest_rate = min(standard_rates, key=lambda x: abs(x - detected_bph))
                
                # Actualizar si la detección es confiable
                error_percent = abs(detected_bph - closest_rate) / closest_rate
                if error_percent < 0.15:  # Dentro del 15% de tasa estándar
                    if self.beat_rate != closest_rate:
                        self.set_beat_rate(closest_rate)
                        print(f"Auto-detectada tasa de beats: {closest_rate} BPH")
    
    def calculate_confidence(self):
        """Calcular métricas de confianza"""
        # Verificación mínima de datos
        if len(self.tick_intervals) < 5:
            self.overall_confidence = 0
            return
        
        # Intervalos recientes para análisis
        recent_intervals = list(self.tick_intervals)[-min(len(self.tick_intervals), 50):]
        recent_amplitudes = list(self.amplitudes)[-min(len(self.amplitudes), 50):]
        
        # Consistencia de intervalos
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
        
        # Consistencia de amplitudes
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
        
        # Proxy de SNR basado en amplitud
        snr_score = min(10, max(0, np.mean(recent_amplitudes) / 5))
        
        # Métrica de confianza combinada
        # Ponderación: 50% consistencia de intervalos, 30% consistencia de amplitudes, 20% SNR
        self.overall_confidence = (0.5 * self.interval_consistency + 
                                  0.3 * self.amplitude_consistency + 
                                  0.2 * snr_score)
        
        # Actualizar confianza en resultados
        self.results['confidence'] = self.overall_confidence
        self.results['quality_metrics']['interval_consistency'] = self.interval_consistency
        self.results['quality_metrics']['amplitude_consistency'] = self.amplitude_consistency
        self.results['quality_metrics']['signal_to_noise'] = snr_score
    
    def reset(self):
        """Resetear todos los datos de análisis"""
        # Resetear detector extremo
        self.tick_detector.reset()
        
        # Limpiar estructuras de datos
        self.tick_times.clear()
        self.tick_intervals.clear()
        self.amplitudes.clear()
        self.beat_errors.clear()
        
        # Resetear métricas
        self.interval_consistency = 0
        self.amplitude_consistency = 0
        self.overall_confidence = 0
        
        # Resetear resultados
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
        
        # Resetear estado
        self.has_analysis_data = False
        self.start_time = time.time()
        
        print("Datos de análisis reseteados")
    
    def analyze_timing(self):
        """Calcular métricas de timegrapher con mejor manejo de señales débiles"""
        if len(self.tick_intervals) < 5:
            return
            
        # Obtener intervalos y amplitudes recientes para análisis
        recent_intervals = list(self.tick_intervals)[-min(len(self.tick_intervals), 50):]
        recent_amplitudes = list(self.amplitudes)[-min(len(self.amplitudes), 50):]
        
        # Eliminar valores atípicos
        if len(recent_intervals) > 10:
            mean_interval = np.mean(recent_intervals)
            std_interval = np.std(recent_intervals)
            
            if std_interval > 0:
                # Mantener intervalos dentro del umbral de la media
                filtered_intervals = []
                filtered_amplitudes = []
                
                for i, interval in enumerate(recent_intervals):
                    if i < len(recent_amplitudes):
                        # Ajustar umbral basado en calidad de señal
                        if self.overall_confidence < 3:
                            # Señal muy débil - ser más permisivo
                            threshold = self.outlier_threshold * 1.5
                        elif self.overall_confidence < 6:
                            # Señal de calidad media
                            threshold = self.outlier_threshold * 1.2
                        else:
                            # Buena señal - usar umbral estándar
                            threshold = self.outlier_threshold
                            
                        if abs(interval - mean_interval) <= threshold * std_interval:
                            filtered_intervals.append(interval)
                            if i < len(recent_amplitudes):
                                filtered_amplitudes.append(recent_amplitudes[i])
                
                if len(filtered_intervals) > 3:
                    recent_intervals = filtered_intervals
                    recent_amplitudes = filtered_amplitudes
        
        # Calcular tasa
        if len(recent_intervals) > 0:
            # Intervalo promedio en segundos
            avg_interval_sec = np.mean(recent_intervals)
            
            # Convertir a BPH (Beats Por Hora)
            if avg_interval_sec > 0:
                beats_per_second = 1 / (avg_interval_sec * 2)  # Cada tick es medio beat
                beats_per_hour = beats_per_second * 3600
                
                # Buscar tasa estándar más cercana
                standard_rates = [18000, 19800, 21600, 25200, 28800, 36000]
                closest_rate = min(standard_rates, key=lambda x: abs(x - beats_per_hour))
                
                # Calcular desviación en segundos por día
                rate_difference = beats_per_hour - closest_rate
                seconds_per_day = (rate_difference / closest_rate) * 86400
                
                # Actualizar resultados
                self.results['rate'] = beats_per_hour
                self.results['seconds_per_day'] = seconds_per_day
                self.results['beat_error'] = self.calculate_beat_error(recent_intervals)
                
                # Estimar amplitud basada en métricas de calidad
                if len(recent_amplitudes) > 0:
                    normalized_amplitude = np.mean(recent_amplitudes) / 100
                    amplitude_estimate = 200 + (normalized_amplitude * 115)  # Mapear a 200-315°
                    self.results['amplitude'] = amplitude_estimate
                
                # Actualizar colecciones de resultados
                self.results['tick_times'] = list(self.tick_times)
                self.results['tick_intervals'] = list(self.tick_intervals)
    
    def calculate_beat_error(self, intervals):
        """Calcular error de beat a partir de intervalos"""
        if len(intervals) < 4:
            return 0
            
        # Agrupar intervalos en pares para detectar asimetría
        beat_errors = []
        for i in range(0, len(intervals) - 1, 2):
            if i + 1 < len(intervals):
                # Cada par de ticks consecutivos forma tick-tock
                tick_interval = intervals[i]
                tock_interval = intervals[i + 1]
                total_interval = tick_interval + tock_interval
                
                if total_interval > 0:
                    # Calcular asimetría - lo perfecto sería 0.5 (intervalos iguales)
                    ratio = tick_interval / total_interval
                    asymmetry = abs(ratio - 0.5) * 2  # Escalar a 0-1
                    beat_errors.append(asymmetry * 100)  # Convertir a porcentaje
        
        # Devolver beat error promedio
        if beat_errors:
            return np.mean(beat_errors)
        return 0
    
    def get_results(self):
        """Obtener resultados actuales del análisis"""
        return self.results
        
    def has_data(self):
        """Comprobar si tenemos datos de análisis"""
        return self.has_analysis_data
        
    def export_results(self, filename):
        """Exportar resultados a CSV"""
        if not self.has_analysis_data:
            print("No hay datos para exportar")
            return False
            
        try:
            # Crear DataFrame de nuestras mediciones
            data = {
                'Tiempo': list(self.tick_times),
                'Intervalo': list(self.tick_intervals),
                'Amplitud': list(self.amplitudes),
            }
            
            # Añadir beat error si está disponible
            if len(self.beat_errors) > 0:
                data['Beat Error'] = list(self.beat_errors)
                
            # Crear DataFrame
            df = pd.DataFrame(data)
            
            # Añadir metadatos
            metadata = pd.DataFrame({
                'Parámetro': ['Tasa de Beat', 'Segundos por día', 'Amplitud Promedio', 'Confianza'],
                'Valor': [
                    f"{self.results['rate']:.0f} BPH",
                    f"{self.results.get('seconds_per_day', 0):.1f} s/d",
                    f"{self.results['amplitude']:.1f}°",
                    f"{self.results['confidence']:.1f}/10"
                ]
            })
            
            # Exportar a CSV
            with open(filename, 'w') as f:
                # Escribir metadatos
                f.write("# Resultados de Análisis de Timegrapher\n")
                metadata.to_csv(f, index=False)
                f.write("\n# Datos de Medición\n")
                df.to_csv(f, index=False)
                
            print(f"Resultados exportados a {filename}")
            return True
            
        except Exception as e:
            print(f"Error exportando resultados: {e}")
            return False
