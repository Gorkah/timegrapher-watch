#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timegrapher Básico - Una versión super simplificada y directa
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd
import queue
import threading
import sys
import time
from scipy import signal
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QSlider, QGroupBox, QGridLayout, QSpinBox, 
                            QCheckBox, QDoubleSpinBox, QFrame)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

# Parámetros globales (simplificados)
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
PLOT_DURATION = 3  # segundos
MAX_AMPLITUDE = 0.5  # Para escala de visualización

# Cola para los datos de audio
audio_queue = queue.Queue()
# Cola para los ticks detectados
tick_queue = queue.Queue()

class AudioThread(threading.Thread):
    """Thread simplificado para captura de audio"""
    
    def __init__(self, device=None, gain=20.0):
        super().__init__()
        self.device = device
        self.gain = gain
        self.running = True
        self.stream = None
        
    def run(self):
        """Ejecutar thread de captura de audio"""
        def callback(indata, frames, time, status):
            """Callback para el stream de audio"""
            if status:
                print(f"Status: {status}")
            
            # Aplicar ganancia y enviar a cola
            audio_data = np.squeeze(indata.copy()) * self.gain
            audio_queue.put(audio_data)
        
        # Iniciar stream de audio
        with sd.InputStream(device=self.device, channels=1, 
                           samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE,
                           callback=callback):
            print(f"Stream de audio iniciado con ganancia {self.gain}x")
            
            # Mantener thread activo
            while self.running:
                time.sleep(0.1)
    
    def stop(self):
        """Detener thread de audio"""
        self.running = False

class SignalPlot(FigureCanvas):
    """Widget simplificado para visualización de señal"""
    
    tick_detected = pyqtSignal(float, float)  # Señal para ticks (timestamp, amplitud)
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Inicializar visualización de señal"""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(SignalPlot, self).__init__(self.fig)
        
        # Buffer para datos de audio
        self.buffer_size = PLOT_DURATION * SAMPLE_RATE
        self.signal_buffer = np.zeros(self.buffer_size)
        
        # Configuración de visualización
        self.axes.set_ylim(-MAX_AMPLITUDE, MAX_AMPLITUDE)
        self.axes.set_xlim(0, PLOT_DURATION)
        self.axes.set_xlabel('Tiempo (s)')
        self.axes.set_ylabel('Amplitud')
        self.axes.set_title('Señal de Audio en Tiempo Real')
        self.axes.grid(True)
        
        # Línea para la señal
        self.line, = self.axes.plot(np.linspace(0, PLOT_DURATION, self.buffer_size), 
                                    self.signal_buffer)
        
        # Marcadores para ticks detectados
        self.tick_markers = []
        
        # Configuración
        self.threshold = 0.05      # Umbral para detección
        self.filter_low = 800      # Hz para filtro paso alto
        self.filter_high = 8000    # Hz para filtro paso bajo
        
        # Ajuste de figura
        self.fig.tight_layout()
    
    def update_plot(self):
        """Actualizar visualización con nuevos datos"""
        # Obtener datos de audio si están disponibles
        data_collected = False
        while not audio_queue.empty():
            try:
                new_data = audio_queue.get_nowait()
                # Actualizar buffer circular
                self.signal_buffer = np.roll(self.signal_buffer, -len(new_data))
                self.signal_buffer[-len(new_data):] = new_data
                data_collected = True
            except queue.Empty:
                break
        
        if not data_collected:
            return
        
        # Procesar señal con filtros básicos
        try:
            # Asegurar valores de filtro válidos
            low_freq = max(10, min(self.filter_low, SAMPLE_RATE/2 - 100))
            high_freq = max(low_freq + 100, min(self.filter_high, SAMPLE_RATE/2 - 1))
            
            # Filtros para aislar ticks
            sos_high = signal.butter(4, low_freq, 'highpass', fs=SAMPLE_RATE, output='sos')
            sos_low = signal.butter(4, high_freq, 'lowpass', fs=SAMPLE_RATE, output='sos')
            
            filtered = signal.sosfilt(sos_high, self.signal_buffer)
            filtered = signal.sosfilt(sos_low, filtered)
            
            # Calcular envolvente
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)
            
            # Suavizar envolvente
            window_size = int(SAMPLE_RATE * 0.001)  # 1ms
            if window_size % 2 == 0:
                window_size += 1
            
            envelope = signal.savgol_filter(envelope, window_size, 2)
            
            # Mostrar envolvente
            self.line.set_ydata(envelope)
            
            # Detectar picos y emitir señal
            peaks, properties = signal.find_peaks(
                envelope, 
                height=self.threshold,
                distance=int(SAMPLE_RATE * 0.1)  # Mínimo 100ms entre picos
            )
            
            # Limpiar marcadores anteriores
            for marker in self.tick_markers:
                marker.remove()
            self.tick_markers = []
            
            # Procesar nuevos picos
            if len(peaks) > 0:
                time_axis = np.linspace(0, PLOT_DURATION, len(self.signal_buffer))
                for i, peak in enumerate(peaks):
                    # Mostrar marcador en gráfico
                    marker = self.axes.axvline(x=time_axis[peak], color='r', alpha=0.7)
                    self.tick_markers.append(marker)
                    
                    # Emitir señal de tick detectado
                    self.tick_detected.emit(time.time(), properties["peak_heights"][i])
                    
                    # También poner en cola para procesamiento
                    tick_queue.put(time.time())
            
        except Exception as e:
            print(f"Error en procesamiento: {e}")
        
        # Redibujar
        self.draw()

class BasicTimegrapherApp(QMainWindow):
    """Aplicación super simplificada para timegrapher"""
    
    def __init__(self):
        super().__init__()
        
        # Estado
        self.running = False
        self.audio_thread = None
        
        # Para análisis
        self.tick_times = []
        self.last_update = time.time()
        self.bph = 0
        self.deviation = 0
        self.beat_error = 0
        
        # Inicialización
        self.init_ui()
        self.setup_audio_devices()
        
        # Timer para actualizaciones
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(50)  # 20 FPS
    
    def init_ui(self):
        """Inicializar UI simplificada"""
        self.setWindowTitle('Timegrapher Básico')
        self.resize(800, 600)
        
        # Widget central
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Área de visualización
        self.signal_plot = SignalPlot(width=8, height=4)
        self.signal_plot.tick_detected.connect(self.on_tick_detected)
        main_layout.addWidget(self.signal_plot)
        
        # Controles básicos
        control_layout = QHBoxLayout()
        
        # Panel de dispositivo y ganancia
        input_group = QGroupBox("Entrada de Audio")
        input_layout = QGridLayout()
        
        # Selector de dispositivo
        self.device_combo = QComboBox()
        input_layout.addWidget(QLabel("Dispositivo:"), 0, 0)
        input_layout.addWidget(self.device_combo, 0, 1)
        
        # Control de ganancia
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(1)
        self.gain_slider.setMaximum(80)
        self.gain_slider.setValue(30)
        self.gain_label = QLabel("Ganancia: 30x")
        input_layout.addWidget(QLabel("Ganancia:"), 1, 0)
        input_layout.addWidget(self.gain_slider, 1, 1)
        input_layout.addWidget(self.gain_label, 1, 2)
        
        # Control de umbral
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(50)
        self.threshold_slider.setValue(5)
        self.threshold_label = QLabel("Umbral: 0.05")
        input_layout.addWidget(QLabel("Umbral:"), 2, 0)
        input_layout.addWidget(self.threshold_slider, 2, 1)
        input_layout.addWidget(self.threshold_label, 2, 2)
        
        input_group.setLayout(input_layout)
        control_layout.addWidget(input_group)
        
        # Panel de filtros
        filter_group = QGroupBox("Filtros")
        filter_layout = QGridLayout()
        
        self.lowfreq_spinbox = QSpinBox()
        self.lowfreq_spinbox.setRange(100, 2000)
        self.lowfreq_spinbox.setValue(800)
        self.lowfreq_spinbox.setSingleStep(50)
        
        self.highfreq_spinbox = QSpinBox()
        self.highfreq_spinbox.setRange(2000, 20000)
        self.highfreq_spinbox.setValue(8000)
        self.highfreq_spinbox.setSingleStep(500)
        
        filter_layout.addWidget(QLabel("Paso Alto (Hz):"), 0, 0)
        filter_layout.addWidget(self.lowfreq_spinbox, 0, 1)
        filter_layout.addWidget(QLabel("Paso Bajo (Hz):"), 1, 0)
        filter_layout.addWidget(self.highfreq_spinbox, 1, 1)
        
        # Botón para aplicar filtros
        self.apply_filter_button = QPushButton("Aplicar Filtros")
        filter_layout.addWidget(self.apply_filter_button, 2, 0, 1, 2)
        
        filter_group.setLayout(filter_layout)
        control_layout.addWidget(filter_group)
        
        # Resultados
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout()
        
        self.rate_label = QLabel("Tasa: -- BPH")
        self.deviation_label = QLabel("Desviación: -- s/d")
        self.error_label = QLabel("Beat Error: --%")
        self.ticks_label = QLabel("Ticks detectados: 0")
        
        results_layout.addWidget(self.rate_label)
        results_layout.addWidget(self.deviation_label)
        results_layout.addWidget(self.error_label)
        results_layout.addWidget(self.ticks_label)
        
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group)
        
        main_layout.addLayout(control_layout)
        
        # Botones de control
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Iniciar")
        self.stop_button = QPushButton("Detener")
        self.reset_button = QPushButton("Reset")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        
        self.stop_button.setEnabled(False)
        
        main_layout.addLayout(button_layout)
        
        # Conectar eventos
        self.gain_slider.valueChanged.connect(self.update_gain)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.apply_filter_button.clicked.connect(self.apply_filters)
        
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.reset_button.clicked.connect(self.reset_analysis)
        
        # Instrucciones
        instructions = QLabel("Coloca el reloj directamente sobre el micrófono. "
                             "Ajusta la ganancia y umbral hasta que los ticks se detecten claramente.")
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)
        
        self.setCentralWidget(central_widget)
    
    def setup_audio_devices(self):
        """Configurar dispositivos de audio disponibles"""
        devices = sd.query_devices()
        input_devices = []
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = f"{i}: {dev['name']}"
                input_devices.append((name, i))
        
        for name, idx in input_devices:
            self.device_combo.addItem(name, idx)
    
    def update_gain(self):
        """Actualizar valor de ganancia"""
        gain = self.gain_slider.value()
        self.gain_label.setText(f"Ganancia: {gain}x")
        
        # Actualizar thread de audio si está en ejecución
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.gain = gain
    
    def update_threshold(self):
        """Actualizar umbral de detección"""
        value = self.threshold_slider.value() / 100
        self.threshold_label.setText(f"Umbral: {value:.2f}")
        self.signal_plot.threshold = value
    
    def apply_filters(self):
        """Aplicar nuevos valores de filtro"""
        low = self.lowfreq_spinbox.value()
        high = self.highfreq_spinbox.value()
        
        # Validar que low < high
        if low >= high:
            print("Error: El valor de paso alto debe ser menor que el de paso bajo")
            self.lowfreq_spinbox.setValue(high - 500)
            return
        
        # Aplicar valores al plot
        self.signal_plot.filter_low = low
        self.signal_plot.filter_high = high
        print(f"Filtros aplicados: {low} - {high} Hz")
    
    def on_tick_detected(self, timestamp, amplitude):
        """Manejar tick detectado desde la visualización"""
        # Almacenar tiempo de tick
        self.tick_times.append(timestamp)
        
        # Limpiar ticks muy antiguos (más de 30 segundos)
        current_time = time.time()
        self.tick_times = [t for t in self.tick_times if current_time - t < 30]
        
        # Actualizar contador
        self.ticks_label.setText(f"Ticks detectados: {len(self.tick_times)}")
    
    def start_analysis(self):
        """Iniciar análisis de audio"""
        if not self.running:
            # Obtener dispositivo y ganancia
            device_idx = self.device_combo.currentData()
            gain = self.gain_slider.value()
            
            # Iniciar captura de audio
            self.audio_thread = AudioThread(device=device_idx, gain=gain)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Actualizar UI
            self.running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.device_combo.setEnabled(False)
    
    def stop_analysis(self):
        """Detener análisis"""
        if self.running:
            # Detener thread de audio
            if self.audio_thread:
                self.audio_thread.stop()
                self.audio_thread.join(timeout=1)
                self.audio_thread = None
            
            # Actualizar UI
            self.running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.device_combo.setEnabled(True)
    
    def reset_analysis(self):
        """Resetear análisis"""
        # Limpiar datos
        self.tick_times = []
        
        # Limpiar buffer de visualización
        self.signal_plot.signal_buffer = np.zeros(self.signal_plot.buffer_size)
        
        # Actualizar etiquetas
        self.rate_label.setText("Tasa: -- BPH")
        self.deviation_label.setText("Desviación: -- s/d")
        self.error_label.setText("Beat Error: --%")
        self.ticks_label.setText("Ticks detectados: 0")
        
        # Limpiar colas
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except:
                pass
        
        while not tick_queue.empty():
            try:
                tick_queue.get_nowait()
            except:
                pass
    
    def analyze_ticks(self):
        """Analizar los ticks detectados"""
        # Si no hay suficientes ticks, no hay nada que analizar
        if len(self.tick_times) < 10:
            return
        
        # Calcular intervalos entre ticks
        intervals = []
        for i in range(1, len(self.tick_times)):
            interval = self.tick_times[i] - self.tick_times[i-1]
            # Solo considerar intervalos razonables (entre 0.05 y 1 segundo)
            if 0.05 < interval < 1.0:
                intervals.append(interval)
        
        if len(intervals) < 5:
            return
        
        # Eliminar outliers (fuera de 2 desviaciones estándar)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        good_intervals = [i for i in intervals if abs(i - mean_interval) < 2 * std_interval]
        
        if not good_intervals:
            return
        
        # Calcular BPH
        avg_interval = np.mean(good_intervals)
        beats_per_second = 1 / avg_interval
        self.bph = beats_per_second * 3600
        
        # Calcular desviación
        standard_rates = [18000, 19800, 21600, 25200, 28800, 36000]
        closest_rate = min(standard_rates, key=lambda x: abs(x - self.bph))
        rate_error = (self.bph - closest_rate) / closest_rate
        self.deviation = rate_error * 86400  # segundos por día
        
        # Calcular beat error (simplificado)
        if len(good_intervals) >= 6:
            # Agrupar intervalos en grupos de 2 (alternan)
            odd_intervals = good_intervals[::2][:min(10, len(good_intervals)//2)]
            even_intervals = good_intervals[1::2][:min(10, len(good_intervals)//2)]
            
            if odd_intervals and even_intervals:
                odd_avg = np.mean(odd_intervals)
                even_avg = np.mean(even_intervals)
                total = odd_avg + even_avg
                
                if total > 0:
                    # Calcular asimetría
                    ratio = odd_avg / total
                    self.beat_error = abs(ratio - 0.5) * 100  # como porcentaje
    
    def update(self):
        """Actualizar periódicamente la UI"""
        # Actualizar visualización
        self.signal_plot.update_plot()
        
        # Solo actualizar análisis cada 0.5 segundos para evitar sobrecarga
        current_time = time.time()
        if current_time - self.last_update > 0.5:
            self.last_update = current_time
            
            # Procesar cualquier tick pendiente
            while not tick_queue.empty():
                try:
                    tick_queue.get_nowait()
                except:
                    pass
            
            # Analizar y mostrar resultados
            self.analyze_ticks()
            
            if self.bph > 0:
                self.rate_label.setText(f"Tasa: {self.bph:.1f} BPH")
                self.deviation_label.setText(f"Desviación: {self.deviation:.1f} s/d")
                
                if self.beat_error > 0:
                    self.error_label.setText(f"Beat Error: {self.beat_error:.1f}%")
    
    def closeEvent(self, event):
        """Manejar cierre de ventana"""
        self.stop_analysis()
        event.accept()

def main():
    """Punto de entrada principal"""
    app = QApplication(sys.argv)
    window = BasicTimegrapherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
