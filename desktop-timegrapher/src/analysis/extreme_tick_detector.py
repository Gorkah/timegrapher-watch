#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector extremo de ticks para señales muy débiles
Usa correlación cruzada y métodos avanzados de procesamiento de señal
"""

import numpy as np
from scipy import signal
import time
from collections import deque
try:
    import pywt
except ImportError:
    pywt = None

class ExtremeTickDetector:
    """Detector de ticks optimizado para señales extremadamente débiles"""
    
    def __init__(self, expected_bph=28800):
        """Inicializar detector con tasa esperada en BPH"""
        self.expected_bph = expected_bph
        self.expected_interval = 3600 / expected_bph  # segundos
        self.expected_tick_interval = self.expected_interval / 2  # cada tick es medio beat
        
        # Opciones de configuración (extremadamente sensibles)
        self.pre_gain = 30.0  # Amplificación brutal para señales muy débiles
        self.use_adaptive_gain = True
        self.max_gain = 40.0
        
        # Buffers para detección
        self.buffer_size = 10  # segundos
        self.sample_rate = 44100
        self.audio_buffer = None
        
        # Almacenamiento de ticks detectados
        self.recent_ticks = deque(maxlen=200)
        self.tick_times = deque(maxlen=5000)
        self.tick_intervals = deque(maxlen=5000)
        self.tick_amplitudes = deque(maxlen=5000)
        
        # Estado de detección
        self.start_time = time.time()
        self.last_tick_time = None
        self.noise_profile = None
        self.tick_template = None
        self.tick_template_quality = 0
        
        # Ajustes dinámicos
        self.dynamic_threshold = 0.2
        self.min_threshold = 0.1
        self.signal_quality = 0
        self.last_detection_time = 0
        
        # Contador para actualización de template
        self.detections_since_update = 0
        self.update_template_every = 50
        
    def process_audio(self, audio_data, sample_rate):
        """Procesar fragmento de audio buscando ticks"""
        self.sample_rate = sample_rate
        
        # Inicializar buffer si es necesario
        if self.audio_buffer is None:
            self.audio_buffer = np.zeros(self.buffer_size * sample_rate)
        
        # Actualizar buffer
        data_len = len(audio_data)
        if data_len < len(self.audio_buffer):
            self.audio_buffer = np.roll(self.audio_buffer, -data_len)
            self.audio_buffer[-data_len:] = audio_data
        else:
            self.audio_buffer = audio_data[-len(self.audio_buffer):]
        
        # Procesar detección
        tick_times, tick_qualities = self.detect_ticks(audio_data, sample_rate)
        
        # Almacenar resultados
        current_time = time.time() - self.start_time
        for i, tick_time in enumerate(tick_times):
            # Convertir tiempo relativo a absoluto
            absolute_tick_time = current_time - (len(audio_data) / sample_rate) + tick_time
            
            # Almacenar tiempo
            self.tick_times.append(absolute_tick_time)
            
            # Calcular y almacenar intervalo
            if len(self.tick_times) > 1:
                interval = (self.tick_times[-1] - self.tick_times[-2])
                self.tick_intervals.append(interval)
            
            # Almacenar calidad/amplitud
            if i < len(tick_qualities):
                self.tick_amplitudes.append(tick_qualities[i] * 100)  # Escalar a 0-100
                
        return tick_times, tick_qualities
    
    def detect_ticks(self, audio_data, sample_rate):
        """Detectar ticks en fragmento de audio, optimizado para señales extremadamente débiles"""
        # Paso 1: Pre-amplificación extrema para señales muy débiles
        signal_level = np.percentile(np.abs(audio_data), 95)
        
        # Ganancia adaptativa - más ganancia para señales más débiles
        if self.use_adaptive_gain:
            if signal_level < 0.01:  # Señal extremadamente débil
                gain = min(self.max_gain, 1.0 / max(0.0005, signal_level))
            elif signal_level < 0.05:  # Señal muy débil
                gain = min(30.0, 0.5 / max(0.001, signal_level))
            elif signal_level < 0.1:  # Señal débil
                gain = min(20.0, 0.3 / max(0.001, signal_level))
            else:  # Señal razonable
                gain = self.pre_gain
        else:
            gain = self.pre_gain
            
        # Aplicar ganancia
        amplified_data = audio_data * gain
        
        # Paso 2: Filtrado multi-etapa
        # Filtrado principal para eliminar bajas frecuencias (respiración, movimiento)
        sos_high = signal.butter(4, 800, 'highpass', fs=sample_rate, output='sos')
        high_passed = signal.sosfilt(sos_high, amplified_data)
        
        # Filtrado de bandas específicas de relojes
        sos_band1 = signal.butter(4, [1000, 2000], 'bandpass', fs=sample_rate, output='sos')
        sos_band2 = signal.butter(4, [2500, 3500], 'bandpass', fs=sample_rate, output='sos')
        sos_band3 = signal.butter(4, [4000, 6000], 'bandpass', fs=sample_rate, output='sos')
        
        band1 = signal.sosfilt(sos_band1, high_passed)
        band2 = signal.sosfilt(sos_band2, high_passed)
        band3 = signal.sosfilt(sos_band3, high_passed)
        
        # Combinación de bandas con enfoque en frecuencias medias
        filtered_data = band1 * 1.0 + band2 * 1.5 + band3 * 0.7
        
        # Paso 3: Denoising wavelet si está disponible
        if pywt is not None:
            filtered_data = self._wavelet_denoise(filtered_data)
        
        # Paso 4: Calcular envolvente
        analytic_signal = signal.hilbert(filtered_data)
        envelope = np.abs(analytic_signal)
        
        # Fase y frecuencia instantánea
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
        instantaneous_freq = np.pad(instantaneous_freq, (0, 1), 'edge')
        
        # Detectar cambios bruscos de frecuencia (transitorios)
        freq_transients = np.abs(np.diff(instantaneous_freq))
        freq_transients = np.pad(freq_transients, (0, 1), 'edge')
        
        # Normalizar
        if np.max(freq_transients) > 0:
            norm_transients = freq_transients / np.max(freq_transients)
        else:
            norm_transients = freq_transients
        
        # Paso 5: Crear señal compuesta para detección
        # Combinar envolvente y transitorios
        if np.max(envelope) > 0:
            norm_envelope = envelope / np.max(envelope)
        else:
            norm_envelope = envelope
            
        # Más peso a transitorios para señales débiles
        composite_signal = norm_envelope * 0.3 + norm_transients * 0.7
        
        # Suavizar brevemente para eliminar ruido de alta frecuencia
        win_length = max(5, int(sample_rate * 0.0005))  # 0.5ms
        if win_length % 2 == 0:
            win_length += 1
        
        try:
            smoothed_signal = signal.savgol_filter(composite_signal, win_length, 2)
        except:
            smoothed_signal = composite_signal
        
        # Paso 6: Detección por correlación si tenemos template
        # Este método es mucho más potente para señales muy débiles
        use_template = self.tick_template is not None and self.tick_template_quality > 0.5
        
        if use_template:
            # Correlación normalizada con el template
            correlation = signal.correlate(smoothed_signal, self.tick_template, mode='valid')
            correlation = correlation / np.max(correlation) if np.max(correlation) > 0 else correlation
            
            # Encontrar picos de correlación
            corr_threshold = max(0.6, self.tick_template_quality * 0.5)
            min_dist = int(sample_rate * self.expected_tick_interval * 0.8)  # 80% del intervalo esperado
            
            corr_peaks, _ = signal.find_peaks(
                correlation,
                height=corr_threshold,
                distance=min_dist
            )
            
            # Ajustar índices para compensar desplazamiento de correlación
            template_half = len(self.tick_template) // 2
            template_peaks = [p + template_half for p in corr_peaks if p + template_half < len(smoothed_signal)]
            
            # Mezclar con detección normal
            primary_peaks = template_peaks
        else:
            primary_peaks = []
        
        # Paso 7: Detección tradicional por umbral (como respaldo)
        # Calcular umbral adaptativo
        signal_mean = np.mean(smoothed_signal)
        signal_std = np.std(smoothed_signal)
        signal_max = np.max(smoothed_signal)
        
        # Umbral primario más estricto
        threshold = max(signal_mean + 2.0 * signal_std, signal_max * 0.4) 
        
        # Umbral secundario más permisivo
        low_threshold = max(signal_mean + 1.0 * signal_std, signal_max * 0.2)
        
        # Mínima distancia entre picos (basado en BPH esperado)
        min_distance = int(sample_rate * self.expected_tick_interval * 0.8)
        
        # Detectar picos primarios
        if not primary_peaks:
            primary_peaks, _ = signal.find_peaks(
                smoothed_signal,
                height=threshold,
                distance=min_distance
            )
        
        # Detectar picos secundarios
        all_peaks, _ = signal.find_peaks(
            smoothed_signal,
            height=low_threshold,
            distance=min_distance // 2
        )
        
        # Paso 8: Validación por patrón temporal
        # Si detectamos suficientes picos primarios, usarlos para predecir otros
        if len(primary_peaks) >= 3:
            # Calcular intervalos entre picos
            intervals = np.diff(primary_peaks) / sample_rate
            median_interval = np.median(intervals)
            
            # Comprobar si el intervalo coincide con tasas esperadas
            std_intervals = [0.1, 0.125, 0.143, 0.167, 0.2]  # 36000 a 18000 BPH
            closest_interval = min(std_intervals, key=lambda x: abs(x - median_interval))
            interval_error = abs(median_interval - closest_interval) / closest_interval
            
            # Si es cerca de un valor estándar, buscar picos adicionales
            if interval_error < 0.25:  # 25% de tolerancia
                expected_interval = closest_interval
                expected_samples = int(expected_interval * sample_rate)
                
                # Encontrar picos adicionales buscando en posiciones esperadas
                validated_peaks = list(primary_peaks)
                
                # Buscar adelante y atrás
                for peak in primary_peaks:
                    # Hacia adelante
                    for i in range(1, 5):
                        expected_pos = peak + i * expected_samples
                        if expected_pos >= len(smoothed_signal):
                            continue
                            
                        # Buscar en ventana alrededor de posición esperada
                        window_size = int(expected_samples * 0.25)  # 25% tolerancia
                        start = max(0, expected_pos - window_size)
                        end = min(len(smoothed_signal), expected_pos + window_size)
                        
                        if start >= end:
                            continue
                            
                        # Encontrar máximo local
                        window_max = np.argmax(smoothed_signal[start:end]) + start
                        
                        # Añadir si supera umbral y no está ya añadido
                        if (smoothed_signal[window_max] > low_threshold and 
                            window_max not in validated_peaks):
                            validated_peaks.append(window_max)
                    
                    # Hacia atrás
                    for i in range(1, 5):
                        expected_pos = peak - i * expected_samples
                        if expected_pos < 0:
                            continue
                            
                        window_size = int(expected_samples * 0.25)
                        start = max(0, expected_pos - window_size)
                        end = min(len(smoothed_signal), expected_pos + window_size)
                        
                        if start >= end:
                            continue
                            
                        window_max = np.argmax(smoothed_signal[start:end]) + start
                        
                        if (smoothed_signal[window_max] > low_threshold and 
                            window_max not in validated_peaks):
                            validated_peaks.append(window_max)
                
                # Usar picos validados por patrón
                peaks = np.array(sorted(validated_peaks))
            else:
                # Si no hay patrón claro, usar todos los picos detectados
                peaks = all_peaks
        else:
            # Pocos picos primarios, usar todos los detectados
            peaks = all_peaks
        
        # Paso 9: Actualizar template de tick si tenemos buenos picos
        # Esto mejora significativamente las detecciones futuras
        if len(primary_peaks) > 3:
            self.detections_since_update += 1
            
            # Actualizar template periódicamente
            if (self.detections_since_update > self.update_template_every or 
                self.tick_template is None):
                
                # Crear template promediando varios ticks detectados
                template_width = int(sample_rate * 0.01)  # 10ms
                templates = []
                
                for peak in primary_peaks:
                    if peak - template_width >= 0 and peak + template_width < len(smoothed_signal):
                        template = smoothed_signal[peak-template_width:peak+template_width]
                        # Normalizar
                        if np.max(template) > 0:
                            template = template / np.max(template)
                            templates.append(template)
                
                if templates:
                    # Promediar templates
                    new_template = np.mean(templates, axis=0)
                    
                    # Calcular calidad basada en consistencia
                    template_similarity = []
                    for t in templates:
                        corr = np.corrcoef(new_template, t)[0, 1]
                        template_similarity.append(corr)
                    
                    template_quality = np.mean(template_similarity)
                    
                    # Actualizar si es mejor que el actual
                    if self.tick_template is None or template_quality > self.tick_template_quality:
                        self.tick_template = new_template
                        self.tick_template_quality = template_quality
                        print(f"Template actualizado. Calidad: {template_quality:.2f}")
                
                self.detections_since_update = 0
        
        # Paso 10: Calcular métricas de calidad para cada pico
        if len(peaks) > 0:
            # Altura del pico
            peak_heights = smoothed_signal[peaks]
            
            # Normalizar alturas
            if np.max(peak_heights) > 0:
                normalized_heights = peak_heights / np.max(peak_heights)
            else:
                normalized_heights = peak_heights
            
            # Consistencia de intervalos
            if len(peaks) > 1:
                intervals = np.diff(peaks) / sample_rate
                if np.mean(intervals) > 0:
                    interval_consistency = 1.0 - min(1.0, np.std(intervals) / np.mean(intervals))
                else:
                    interval_consistency = 0.0
            else:
                interval_consistency = 0.0
            
            # Calidad combinada
            qualities = 0.7 * normalized_heights
            if len(peaks) > 1:
                qualities = qualities + (0.3 * interval_consistency)
            
            # Bonus para picos primarios
            for i, peak in enumerate(peaks):
                if peak in primary_peaks:
                    qualities[i] = min(1.0, qualities[i] * 1.3)
                    
            # Si usamos template, dar bonus adicional
            if use_template:
                qualities = qualities * 1.2
        else:
            qualities = np.array([])
        
        # Convertir índices a tiempos
        tick_times = peaks / sample_rate
        
        return tick_times, qualities
    
    def _wavelet_denoise(self, data, wavelet='db4', level=3):
        """Aplicar denoising wavelet para mejorar la señal"""
        try:
            # Descomposición wavelet
            coeffs = pywt.wavedec(data, wavelet, level=level)
            
            # Umbralizado de coeficientes (excepto aproximación)
            for i in range(1, len(coeffs)):
                abs_coeffs = np.abs(coeffs[i])
                if len(abs_coeffs) > 0:
                    # Umbral adaptativo
                    threshold = np.median(abs_coeffs) / 0.6745 * np.sqrt(2 * np.log(len(data)))
                    coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # Reconstrucción
            return pywt.waverec(coeffs, wavelet)
        except Exception as e:
            print(f"Error en wavelet denoising: {e}")
            return data
    
    def set_expected_bph(self, bph):
        """Establecer tasa de pulsos esperada en BPH"""
        self.expected_bph = bph
        self.expected_interval = 3600 / bph  # segundos
        self.expected_tick_interval = self.expected_interval / 2
    
    def reset(self):
        """Resetear el detector"""
        self.tick_times.clear()
        self.tick_intervals.clear()
        self.tick_amplitudes.clear()
        self.recent_ticks.clear()
        self.start_time = time.time()
        self.last_tick_time = None
        self.detections_since_update = 0
        print("Detector reseteado")
    
    def get_results(self):
        """Obtener resultados para análisis"""
        return {
            'tick_times': list(self.tick_times),
            'tick_intervals': list(self.tick_intervals),
            'tick_amplitudes': list(self.tick_amplitudes)
        }
