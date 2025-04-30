#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timegrapher de Precisión - Versión que garantiza correspondencia entre visualización y métricas
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
                            QCheckBox, QDoubleSpinBox, QFrame, QProgressBar)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

# No es necesaria la importación adicional ya que QVBoxLayout está incluido arriba

# Parámetros globales
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
PLOT_DURATION = 3  # segundos
MAX_AMPLITUDE = 0.5  # Para escala de visualización

# Tasas estándar para relojes mecánicos
STANDARD_RATES = [18000, 19800, 21600, 25200, 28800, 36000]

# Colas para comunicación entre hilos
audio_queue = queue.Queue()

class AudioThread(threading.Thread):
    """Thread para captura de audio continua"""
    
    def __init__(self, device=None, gain=20.0):
        super().__init__()
        self.device = device
        self.gain = gain
        self.running = True
        
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
    """Widget para visualizar la señal de audio"""
    
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
        self.peak_timestamps = []  # Lista para almacenar tiempos exactos de los picos
        
        # Configuración
        self.threshold = 0.05      # Umbral para detección
        self.filter_low = 800      # Hz para filtro paso alto
        self.filter_high = 8000    # Hz para filtro paso bajo
        
        # Para cálculos - ventana extremadamente amplia para estabilidad máxima
        self.last_process_time = time.time()
        self.detection_intervals = []  # Para almacenar intervalos entre detecciones
        self.historical_intervals = []  # Histórico de intervalos para promedios más estables
        self.max_interval_history = 500  # Mantener hasta 500 intervalos para cálculos (varios minutos)
        self.last_tick_time = None
        
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
            return False
        
        # Procesar señal con filtros
        try:
            # Verificar valores de filtro
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
            
            # Limpiar marcadores anteriores
            for marker in self.tick_markers:
                marker.remove()
            self.tick_markers = []
            
            # Detectar picos 
            peaks, properties = signal.find_peaks(
                envelope, 
                height=self.threshold,
                distance=int(SAMPLE_RATE * 0.1)  # Mínimo 100ms entre picos
            )
            
            # Procesar picos detectados
            current_time = time.time()
            time_axis = np.linspace(0, PLOT_DURATION, len(self.signal_buffer))
            
            # Almacenar los tiempos de los picos para visualización
            self.peak_timestamps = []
            
            for i, peak in enumerate(peaks):
                # Mostrar marcador en gráfico
                marker = self.axes.axvline(x=time_axis[peak], color='r', alpha=0.7)
                self.tick_markers.append(marker)
                
                # Calcular timestamp preciso del pico
                peak_time = current_time - (PLOT_DURATION - time_axis[peak])
                self.peak_timestamps.append(peak_time)
                
                # Calcular intervalo solo si hay un pico anterior
                if self.last_tick_time is not None:
                    interval = peak_time - self.last_tick_time
                    # Solo considerar intervalos razonables (0.05s a 1s)
                    if 0.05 < interval < 1.0:
                        self.detection_intervals.append(interval)
                
                self.last_tick_time = peak_time
                
                # Emitir señal con información del tick
                self.tick_detected.emit(peak_time, properties["peak_heights"][i])
            
            # Redibujar canvas
            self.draw()
            return len(peaks) > 0
            
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            return False
    
    def get_detection_data(self):
        """Obtener datos de detección para cálculos"""
        # Actualizar datos para análisis
        current_time = time.time()
        if current_time - self.last_process_time > 5:
            # Añadir los nuevos intervalos al histórico
            self.historical_intervals.extend(self.detection_intervals)
            
            # Limitar el tamaño del histórico para evitar desbordamiento de memoria
            if len(self.historical_intervals) > self.max_interval_history:
                self.historical_intervals = self.historical_intervals[-self.max_interval_history:]
                
            # Reiniciar intervalos detectados recientemente para evitar duplicados
            self.detection_intervals = []
            self.last_process_time = current_time
        
        # Devolver tanto intervalos recientes como históricos para cálculos
        return {
            'recent_intervals': self.detection_intervals,  # Para visualización
            'historical_intervals': self.historical_intervals,  # Para cálculos estables
            'timestamps': self.peak_timestamps
        }
    
    def reset_detection(self):
        """Resetear detección"""
        self.detection_intervals = []
        self.historical_intervals = []
        self.peak_timestamps = []
        self.last_tick_time = None
        self.signal_buffer = np.zeros(self.buffer_size)
        
        # Limpiar marcadores
        for marker in self.tick_markers:
            marker.remove()
        self.tick_markers = []
        
        # Actualizar línea
        self.line.set_ydata(self.signal_buffer)
        self.draw()

class PrecisionTimegrapherApp(QMainWindow):
    """Aplicación principal del timegrapher de precisión"""
    
    def __init__(self):
        super().__init__()
        
        # Estado
        self.running = False
        self.audio_thread = None
        
        # Para análisis
        self.bph = 0
        self.deviation = 0
        self.beat_error = 0
        self.tick_count = 0
        
        # Historial de métricas para estabilidad extrema
        self.bph_history = deque(maxlen=50)         # 50 mediciones para BPH
        self.deviation_history = deque(maxlen=100)  # 100 mediciones para desviación (ultra-estable)
        self.beat_error_history = deque(maxlen=50)   # 50 mediciones para beat error
        
        # Contadores para actualización
        self.update_counter = 0
        self.metrics_update_interval = 10  # Sólo actualizar métricas cada 10 ciclos
        
        # Inicialización
        self.init_ui()
        self.setup_audio_devices()
        
        # Timer para actualizaciones
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(50)  # 20 FPS
    
    def init_ui(self):
        """Inicializar interfaz de usuario"""
        self.setWindowTitle('Timegrapher de Precisión')
        self.resize(800, 600)
        
        # Widget central
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Área de visualización
        self.signal_plot = SignalPlot(width=8, height=4)
        self.signal_plot.tick_detected.connect(self.on_tick_detected)
        main_layout.addWidget(self.signal_plot)
        
        # Controles principales
        control_layout = QHBoxLayout()
        
        # Panel de dispositivo y ganancia
        input_group = QGroupBox("Entrada de Audio")
        input_layout = QGridLayout()
        
        # Selector de dispositivo
        self.device_combo = QComboBox()
        input_layout.addWidget(QLabel("Dispositivo:"), 0, 0)
        input_layout.addWidget(self.device_combo, 0, 1, 1, 2)
        
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
        
        # Barra de estado
        self.status_bar = QProgressBar()
        self.status_bar.setRange(0, 100)
        self.status_bar.setValue(0)
        self.status_bar.setFormat("Esperando datos...")
        main_layout.addWidget(self.status_bar)
        
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
        
        # Verificar valores válidos
        if low < 50 or high > 22050 or high - low < 100:
            print("Valores de filtro inválidos")
            return
        
        # Aplicar valores al plot
        self.signal_plot.filter_low = low
        self.signal_plot.filter_high = high
        print(f"Filtros aplicados: {low} - {high} Hz")
        
        # Resetear análisis para evitar mezclar datos con diferentes configuraciones
        self.reset_analysis(keep_ui_state=True)
    
    def on_tick_detected(self, timestamp, amplitude):
        """Manejar tick detectado desde la visualización"""
        self.tick_count += 1
        self.ticks_label.setText(f"Ticks detectados: {self.tick_count}")
        
        # Actualizar barra de estado
        if self.tick_count < 10:
            self.status_bar.setValue(self.tick_count * 10)
            self.status_bar.setFormat(f"Recolectando datos: {self.tick_count}/10 ticks")
        else:
            self.status_bar.setValue(100)
            self.status_bar.setFormat("Analizando datos")
    
    def analyze_data(self):
        """Analizar datos de ticks detectados"""
        # Obtener datos actuales de detección
        detection_data = self.signal_plot.get_detection_data()
        recent_intervals = detection_data['recent_intervals']
        historical_intervals = detection_data['historical_intervals']
        
        # Combinar intervalos recientes con algunos históricos para análisis
        # Priorizamos los intervalos recientes pero incluimos históricos para estabilidad
        all_intervals = recent_intervals + historical_intervals
        
        # Verificar si hay suficientes datos
        if len(all_intervals) < 10:  # Aumentado para más estabilidad
            return False
        
        # Filtrar para eliminar outliers
        mean_interval = np.mean(all_intervals)
        std_interval = np.std(all_intervals)
        
        # Solo considerar intervalos dentro de 2 desviaciones estándar
        good_intervals = [i for i in all_intervals if abs(i - mean_interval) < 2 * std_interval]
        
        if len(good_intervals) < 8:  # Aumentado para más estabilidad
            return False
        
        # Calcular BPH con todos los datos buenos disponibles
        avg_interval = np.mean(good_intervals)
        if avg_interval > 0:
            beats_per_second = 1 / avg_interval
            current_bph = beats_per_second * 3600
            
            # Añadir al historial para suavizado
            self.bph_history.append(current_bph)
            
            # Calcular BPH como promedio del historial
            self.bph = np.mean(self.bph_history)
            
            # Encontrar tasa estándar más cercana
            closest_rate = min(STANDARD_RATES, key=lambda x: abs(x - self.bph))
            
            # Calcular desviación
            rate_error = (self.bph - closest_rate) / closest_rate
            current_deviation = rate_error * 86400  # segundos por día
            
            # Añadir al historial para suavizado extremo (evitar fluctuaciones)
            self.deviation_history.append(current_deviation)
            
            # Media extremadamente amortiguada para eliminar fluctuaciones
            # Usamos filtrado por mediana primero y luego promedio ponderado con mayor peso en el centro
            if len(self.deviation_history) >= 5:
                # Primero aplicamos un filtro de mediana para eliminar valores atípicos extremos
                median_filtered = signal.medfilt(np.array(self.deviation_history), kernel_size=5)
                
                # Para desviación extremadamente estable, damos más peso al centro del histórico
                # Esto evita que los nuevos valores o los muy antiguos afecten demasiado
                if len(median_filtered) > 20:
                    # Creamos pesos con forma de campana (más peso en el centro)
                    center = len(median_filtered) // 2
                    weights = np.exp(-0.01 * (np.arange(len(median_filtered)) - center)**2)
                    self.deviation = np.average(median_filtered, weights=weights)
                else:
                    # Si aún no hay suficientes muestras, promedio simple
                    self.deviation = np.mean(median_filtered)
        else:
            return False
        
        # Calcular beat error (requiere suficientes datos) con mayor precisión
        if len(good_intervals) >= 12:  # Necesitamos más datos para un buen beat error
            # Agrupar intervalos alternos
            odd_intervals = good_intervals[::2][:min(20, len(good_intervals)//2)]
            even_intervals = good_intervals[1::2][:min(20, len(good_intervals)//2)]
            
            if odd_intervals and even_intervals:
                odd_avg = np.mean(odd_intervals)
                even_avg = np.mean(even_intervals)
                total = odd_avg + even_avg
                
                if total > 0:
                    # Calcular asimetría
                    ratio = odd_avg / total
                    current_beat_error = abs(ratio - 0.5) * 200  # como porcentaje
                    
                    # Añadir al historial para suavizado
                    self.beat_error_history.append(current_beat_error)
                    self.beat_error = np.mean(self.beat_error_history)
        
        return True
    
    def update_metrics_display(self):
        """Actualizar visualización de métricas con estabilización"""
        # Sólo actualizar el display cada cierto número de ciclos para evitar fluctuaciones visuales
        self.update_counter += 1
        
        if self.update_counter >= self.metrics_update_interval:
            # Redondear valores a 1 decimal para mayor estabilidad visual
            rate = round(self.bph / 10) * 10  # Redondear a decenas
            dev = round(self.deviation, 1)    # 1 decimal para desviación
            err = round(self.beat_error, 1)   # 1 decimal para beat error
            
            self.rate_label.setText(f"Tasa: {rate:.0f} BPH")
            self.deviation_label.setText(f"Desviación: {dev:.1f} s/d")
            self.error_label.setText(f"Beat Error: {err:.1f}%")
            
            self.update_counter = 0
        
    def start_analysis(self):
        """Iniciar análisis de audio"""
        if not self.running:
            # Obtener dispositivo y ganancia
            device_idx = self.device_combo.currentData()
            gain = self.gain_slider.value()
            
            # Resetear análisis
            self.reset_analysis(keep_ui_state=True)
            
            # Iniciar captura de audio
            self.audio_thread = AudioThread(device=device_idx, gain=gain)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Actualizar UI
            self.running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.device_combo.setEnabled(False)
            
            self.status_bar.setValue(0)
            self.status_bar.setFormat("Esperando ticks...")
    
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
            
            self.status_bar.setFormat("Análisis detenido")
    
    def reset_analysis(self, keep_ui_state=False):
        """Resetear análisis"""
        # Reiniciar visualización
        self.signal_plot.reset_detection()
        
        # Reiniciar métricas
        self.bph = 0
        self.deviation = 0
        self.beat_error = 0
        self.tick_count = 0
        
        # Actualizar etiquetas
        self.rate_label.setText("Tasa: -- BPH")
        self.deviation_label.setText("Desviación: -- s/d")
        self.error_label.setText("Beat Error: --%")
        self.ticks_label.setText("Ticks detectados: 0")
        
        # Limpiar cola de audio
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except:
                pass
        
        # Actualizar barra de estado
        if not keep_ui_state:
            self.status_bar.setValue(0)
            self.status_bar.setFormat("Esperando datos...")
    
    def update(self):
        """Actualizar periódicamente la UI con estabilización"""
        # Actualizar visualización
        ticks_detected = self.signal_plot.update_plot()
        
        # Analizar datos cuando haya suficientes ticks
        if self.tick_count >= 20:  # Requerir más ticks antes de mostrar métricas
            if self.analyze_data():
                # Limitar frecuencia de actualización visual para evitar fluctuaciones
                self.update_metrics_display()
    
    def closeEvent(self, event):
        """Manejar cierre de ventana"""
        self.stop_analysis()
        event.accept()

def main():
    """Función principal"""
    app = QApplication(sys.argv)
    window = PrecisionTimegrapherApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
