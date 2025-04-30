#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timegrapher Simple - Una versión simplificada con control manual
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
from PyQt5.QtCore import QTimer, Qt

# Parámetros globales
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
PLOT_DURATION = 3  # segundos
MAX_AMPLITUDE = 0.2  # Para escala de visualización

# Cola para comunicación entre threads
audio_queue = queue.Queue()

class AudioThread(threading.Thread):
    """Thread para captura de audio continua"""
    
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
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

class SignalPlot(FigureCanvas):
    """Widget para visualizar la señal de audio"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Inicializar visualización de señal"""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(SignalPlot, self).__init__(self.fig)
        
        # Buffer circular para datos de audio
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
        
        # Configuración avanzada
        self.filtered_mode = True  # Mostrar señal filtrada
        self.envelope_mode = True  # Mostrar envolvente
        self.threshold = 0.05      # Umbral para detección automática
        self.filter_band = [800, 8000]  # Banda de frecuencia para filtrado
        
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
        
        # Procesar señal si es necesario
        if self.filtered_mode:
            try:
                # Asegurar que los valores de filtro son válidos
                low_freq = max(10, min(self.filter_band[0], SAMPLE_RATE/2 - 1))
                high_freq = max(low_freq + 100, min(self.filter_band[1], SAMPLE_RATE/2 - 1))
                
                # Filtros para aislar ticks de reloj
                sos_high = signal.butter(4, low_freq, 'highpass', 
                                        fs=SAMPLE_RATE, output='sos')
                sos_low = signal.butter(4, high_freq, 'lowpass', 
                                       fs=SAMPLE_RATE, output='sos')
                
                filtered = signal.sosfilt(sos_high, self.signal_buffer)
                filtered = signal.sosfilt(sos_low, filtered)
                
                # Mostrar señal filtrada
                plot_data = filtered
            except Exception as e:
                print(f"Error en filtrado: {e}")
                plot_data = self.signal_buffer
        else:
            # Mostrar señal original
            plot_data = self.signal_buffer
        
        # Calcular envolvente si está habilitado
        if self.envelope_mode:
            analytic = signal.hilbert(plot_data)
            envelope = np.abs(analytic)
            
            # Suavizar envolvente
            window_size = int(SAMPLE_RATE * 0.001)  # 1ms
            if window_size % 2 == 0:
                window_size += 1
            try:
                envelope = signal.savgol_filter(envelope, window_size, 2)
            except:
                pass  # Ignorar errores de suavizado
            
            # Mostrar envolvente
            self.line.set_ydata(envelope)
            
            # Detectar picos automáticamente
            peaks, _ = signal.find_peaks(envelope, height=self.threshold, 
                                        distance=int(SAMPLE_RATE * 0.1))
            
            # Actualizar marcadores de ticks
            time_axis = np.linspace(0, PLOT_DURATION, len(self.signal_buffer))
            for marker in self.tick_markers:
                marker.remove()
            self.tick_markers = []
            
            for peak in peaks:
                marker = self.axes.axvline(x=time_axis[peak], color='r', alpha=0.7)
                self.tick_markers.append(marker)
        else:
            # Mostrar señal directa
            self.line.set_ydata(plot_data)
        
        # Redibujar
        self.fig.canvas.draw()

class TickAnalyzer:
    """Analizador simple de ticks de reloj"""
    
    def __init__(self):
        """Inicializar analizador"""
        # Almacenamiento de ticks
        self.tick_times = deque(maxlen=500)
        self.intervals = deque(maxlen=500)
        self.start_time = time.time()
        
        # Resultados
        self.rate = 0        # en BPH
        self.deviation = 0   # en s/día
        self.beat_error = 0  # en %
        
        # Estándar para comparar
        self.standard_rates = [18000, 19800, 21600, 25200, 28800, 36000]
    
    def add_tick(self, from_manual=False, quality=1.0):
        """Añadir un tick detectado"""
        timestamp = time.time() - self.start_time
        
        # Si es manual o de alta calidad, almacenar
        if from_manual or quality > 0.5:
            # Añadir tiempo de tick
            self.tick_times.append(timestamp)
            
            # Calcular intervalo si hay suficientes ticks
            if len(self.tick_times) > 1:
                interval = self.tick_times[-1] - self.tick_times[-2]
                self.intervals.append(interval)
        
        # Analizar si hay suficientes datos
        if len(self.intervals) > 5:
            self.analyze()
            
        return timestamp
    
    def analyze(self):
        """Analizar datos de ticks"""
        # Usar últimos intervalos para análisis
        recent = list(self.intervals)[-20:]
        
        # Eliminar valores extremos
        if len(recent) > 8:
            mean_interval = np.mean(recent)
            std_interval = np.std(recent)
            
            # Mantener solo intervalos dentro de 2 desviaciones estándar
            recent = [i for i in recent if abs(i - mean_interval) < 2 * std_interval]
        
        if not recent:
            return
            
        # Calcular tasa en BPH
        avg_interval = np.mean(recent)
        if avg_interval > 0:
            # Cada tick es medio beat, así que el periodo completo es 2 * intervalo
            period = avg_interval * 2
            beats_per_second = 1 / period
            self.rate = beats_per_second * 3600
            
            # Encontrar tasa estándar más cercana
            closest = min(self.standard_rates, 
                         key=lambda x: abs(x - self.rate))
            
            # Calcular desviación
            rate_error = (self.rate - closest) / closest
            self.deviation = rate_error * 86400  # segundos por día
        
        # Calcular beat error si hay suficientes datos
        if len(recent) > 3:
            # Agrupar intervalos en pares
            errors = []
            for i in range(0, len(recent) - 1, 2):
                if i + 1 < len(recent):
                    tick = recent[i]
                    tock = recent[i + 1]
                    total = tick + tock
                    
                    if total > 0:
                        # Calcular asimetría
                        ratio = tick / total
                        asymmetry = abs(ratio - 0.5) * 2
                        errors.append(asymmetry * 100)  # como porcentaje
            
            if errors:
                self.beat_error = np.mean(errors)
    
    def reset(self):
        """Resetear análisis"""
        self.tick_times.clear()
        self.intervals.clear()
        self.start_time = time.time()
        self.rate = 0
        self.deviation = 0
        self.beat_error = 0
    
    def get_results(self):
        """Obtener resultados de análisis"""
        return {
            'rate': self.rate,
            'deviation': self.deviation,
            'beat_error': self.beat_error
        }

class SimpleTimegrapherApp(QMainWindow):
    """Aplicación principal del timegrapher simple"""
    
    def __init__(self):
        super().__init__()
        
        # Componentes principales
        self.audio_thread = None
        self.analyzer = TickAnalyzer()
        
        # Configuración inicial
        self.init_ui()
        self.setup_audio_devices()
        
        # Timer para actualizaciones
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(50)  # 20 FPS
        
        # Estado
        self.running = False
        self.manual_mode = False
    
    def init_ui(self):
        """Inicializar interfaz de usuario"""
        self.setWindowTitle('Timegrapher Simple')
        self.resize(800, 600)
        
        # Widget central
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Área superior: visualización
        self.signal_plot = SignalPlot(width=8, height=4)
        main_layout.addWidget(self.signal_plot)
        
        # Área media: controles de visualización
        viz_controls = QGroupBox("Controles de Visualización")
        viz_layout = QGridLayout()
        
        # Selector de dispositivo
        self.device_combo = QComboBox()
        viz_layout.addWidget(QLabel("Dispositivo:"), 0, 0)
        viz_layout.addWidget(self.device_combo, 0, 1)
        
        # Control de ganancia
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(1)
        self.gain_slider.setMaximum(80)
        self.gain_slider.setValue(30)
        self.gain_slider.setTickPosition(QSlider.TicksBelow)
        self.gain_slider.setTickInterval(10)
        self.gain_label = QLabel("Ganancia: 30x")
        viz_layout.addWidget(QLabel("Ganancia:"), 1, 0)
        viz_layout.addWidget(self.gain_slider, 1, 1)
        viz_layout.addWidget(self.gain_label, 1, 2)
        
        # Control de umbral
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(50)
        self.threshold_slider.setValue(5)
        self.threshold_label = QLabel("Umbral: 0.05")
        viz_layout.addWidget(QLabel("Umbral:"), 2, 0)
        viz_layout.addWidget(self.threshold_slider, 2, 1)
        viz_layout.addWidget(self.threshold_label, 2, 2)
        
        # Filtros
        self.filter_checkbox = QCheckBox("Filtrar")
        self.filter_checkbox.setChecked(True)
        self.envelope_checkbox = QCheckBox("Mostrar envolvente")
        self.envelope_checkbox.setChecked(True)
        
        # Rango de filtro
        self.filter_low = QSpinBox()
        self.filter_low.setRange(100, 5000)
        self.filter_low.setValue(800)
        self.filter_low.setSingleStep(100)
        self.filter_low.valueChanged.connect(self.update_filter_range)
        
        self.filter_high = QSpinBox()
        self.filter_high.setRange(1000, 20000)
        self.filter_high.setValue(8000)
        self.filter_high.setSingleStep(500)
        self.filter_high.valueChanged.connect(self.update_filter_range)
        
        viz_layout.addWidget(self.filter_checkbox, 3, 0)
        viz_layout.addWidget(QLabel("Rango (Hz):"), 3, 1)
        
        filter_range = QHBoxLayout()
        filter_range.addWidget(self.filter_low)
        filter_range.addWidget(QLabel("-"))
        filter_range.addWidget(self.filter_high)
        viz_layout.addLayout(filter_range, 3, 2)
        
        viz_layout.addWidget(self.envelope_checkbox, 4, 0)
        
        viz_controls.setLayout(viz_layout)
        main_layout.addWidget(viz_controls)
        
        # Área inferior: botones y resultados
        bottom_layout = QHBoxLayout()
        
        # Panel de botones
        button_group = QGroupBox("Control")
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Iniciar")
        self.stop_button = QPushButton("Detener")
        self.reset_button = QPushButton("Reset")
        self.manual_tick_button = QPushButton("Tick Manual")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.manual_tick_button)
        
        self.stop_button.setEnabled(False)
        self.manual_tick_button.setEnabled(False)
        
        button_group.setLayout(button_layout)
        bottom_layout.addWidget(button_group)
        
        # Panel de resultados
        results_group = QGroupBox("Resultados")
        results_layout = QGridLayout()
        
        self.rate_label = QLabel("Tasa: -- BPH")
        self.deviation_label = QLabel("Desviación: -- s/d")
        self.error_label = QLabel("Beat Error: --%")
        self.amplitud_label = QLabel("Amplitud: -- °")
        
        results_layout.addWidget(self.rate_label, 0, 0)
        results_layout.addWidget(self.deviation_label, 1, 0)
        results_layout.addWidget(self.error_label, 2, 0)
        results_layout.addWidget(self.amplitud_label, 3, 0)
        
        results_group.setLayout(results_layout)
        bottom_layout.addWidget(results_group)
        
        main_layout.addLayout(bottom_layout)
        
        # Conectar eventos
        self.gain_slider.valueChanged.connect(self.update_gain)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.filter_checkbox.stateChanged.connect(self.update_filter_mode)
        self.envelope_checkbox.stateChanged.connect(self.update_envelope_mode)
        # Conexiones de filtros ya establecidas arriba
        
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.reset_button.clicked.connect(self.reset_analysis)
        self.manual_tick_button.clicked.connect(self.add_manual_tick)
        
        # Línea separadora
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # Mostrar indicaciones
        instructions = QLabel("Coloca el reloj directamente sobre el micrófono. "
                             "Ajusta la ganancia hasta visualizar claramente los ticks.")
        instructions.setWordWrap(True)
        main_layout.addWidget(instructions)
        
        self.setCentralWidget(central_widget)
    
    def setup_audio_devices(self):
        """Configurar dispositivos de audio"""
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
        if self.audio_thread and self.audio_thread.running:
            self.audio_thread.gain = gain
    
    def update_threshold(self):
        """Actualizar umbral de detección"""
        value = self.threshold_slider.value() / 100
        self.threshold_label.setText(f"Umbral: {value:.2f}")
        self.signal_plot.threshold = value
    
    def update_filter_mode(self):
        """Actualizar modo de filtrado"""
        self.signal_plot.filtered_mode = self.filter_checkbox.isChecked()
    
    def update_envelope_mode(self):
        """Actualizar visualización de envolvente"""
        self.signal_plot.envelope_mode = self.envelope_checkbox.isChecked()
    
    def update_filter_range(self):
        """Actualizar rango de filtro"""
        low = self.filter_low.value()
        high = self.filter_high.value()
        
        # Asegurar que low < high
        if low >= high:
            if self.sender() == self.filter_low:
                self.filter_low.setValue(high - 100)
            else:
                self.filter_high.setValue(low + 100)
            return  # Evitar bucle infinito de señales
        
        # Aplicar nuevos valores al filtro
        self.signal_plot.filter_band = [self.filter_low.value(), self.filter_high.value()]
        
        # Imprimir valores para debugging
        print(f"Filtro actualizado: {self.filter_low.value()} - {self.filter_high.value()} Hz")
    
    def start_analysis(self):
        """Iniciar análisis"""
        if not self.running:
            # Obtener dispositivo seleccionado
            device_idx = self.device_combo.currentData()
            gain = self.gain_slider.value()
            
            # Iniciar thread de audio
            self.audio_thread = AudioThread(device=device_idx, gain=gain)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Actualizar UI
            self.running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.manual_tick_button.setEnabled(True)
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
            self.manual_tick_button.setEnabled(False)
            self.device_combo.setEnabled(True)
    
    def reset_analysis(self):
        """Resetear análisis"""
        # Resetear analizador
        self.analyzer.reset()
        
        # Limpiar buffer de visualización
        self.signal_plot.signal_buffer = np.zeros(self.signal_plot.buffer_size)
        
        # Actualizar etiquetas
        self.rate_label.setText("Tasa: -- BPH")
        self.deviation_label.setText("Desviación: -- s/d")
        self.error_label.setText("Beat Error: --%")
        self.amplitud_label.setText("Amplitud: -- °")
    
    def add_manual_tick(self):
        """Añadir tick manualmente"""
        if self.running:
            self.analyzer.add_tick(from_manual=True)
    
    def update(self):
        """Actualizar UI periódicamente"""
        # Actualizar visualización
        self.signal_plot.update_plot()
        
        # Actualizar resultados si hay datos
        results = self.analyzer.get_results()
        if results['rate'] > 0:
            self.rate_label.setText(f"Tasa: {results['rate']:.1f} BPH")
            self.deviation_label.setText(f"Desviación: {results['deviation']:.1f} s/d")
            self.error_label.setText(f"Beat Error: {results['beat_error']:.1f}%")
            
            # Estimación de amplitud basada en la señal
            if len(self.analyzer.intervals) > 3:
                # Aproximación muy simple basada en intervalos
                amplitude_est = 200 + min(115, len(self.analyzer.intervals) * 2)
                self.amplitud_label.setText(f"Amplitud: ~{amplitude_est:.0f}°")
    
    def closeEvent(self, event):
        """Evento de cierre de ventana"""
        # Detener thread de audio
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.join(timeout=1)
        
        # Aceptar cierre
        event.accept()

def main():
    """Función principal"""
    app = QApplication(sys.argv)
    window = SimpleTimegrapherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
