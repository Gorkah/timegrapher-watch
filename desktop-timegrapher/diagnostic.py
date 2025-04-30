#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramienta de diagnóstico para detectar problemas con la entrada de audio
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import argparse
import queue
import threading
import time

# Configuración
SAMPLE_RATE = 44100  # Hz
BUFFER_SIZE = 4410   # 100ms por buffer
DISPLAY_TIME = 5     # segundos a mostrar en pantalla
PRE_GAIN = 10.0      # Amplificación previa
MAX_AMPLITUDE = 0.1  # Amplitud máxima para visualización

# Cola para comunicación entre threads
q = queue.Queue()

# Estado global
detected_peaks = []
peak_times = []
signal_level = 0
noise_level = 0
snr = 0

def audio_callback(indata, frames, time, status):
    """Función de callback para la entrada de audio"""
    if status:
        print(f"Estado: {status}")
    
    # Poner datos en la cola
    q.put(indata.copy())

def process_audio():
    """Procesa el audio de la cola"""
    global detected_peaks, peak_times, signal_level, noise_level, snr
    
    # Inicializar buffer
    buffer = np.zeros((DISPLAY_TIME * SAMPLE_RATE,))
    
    while True:
        try:
            # Obtener datos de la cola
            data = q.get(timeout=1)
            data = data.flatten()
            
            # Aplicar pre-amplificación
            data = data * PRE_GAIN
            
            # Actualizar buffer (roll)
            buffer = np.roll(buffer, -len(data))
            buffer[-len(data):] = data
            
            # Análisis de la señal
            # 1. Filtrado
            sos_high = signal.butter(4, 800, 'highpass', fs=SAMPLE_RATE, output='sos')
            sos_band = signal.butter(6, [1000, 8000], 'bandpass', fs=SAMPLE_RATE, output='sos')
            
            filtered = signal.sosfilt(sos_high, buffer)
            filtered = signal.sosfilt(sos_band, filtered)
            
            # 2. Envolvente
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)
            
            # 3. Suavizado
            win_length = int(SAMPLE_RATE * 0.001)  # 1ms
            if win_length % 2 == 0:
                win_length += 1
            envelope_smooth = signal.savgol_filter(envelope, win_length, 2)
            
            # 4. Detección de picos
            # Calcular nivel de señal y ruido
            signal_level = np.percentile(envelope_smooth, 95)
            noise_level = np.percentile(envelope_smooth, 30)
            
            if noise_level > 0:
                snr = signal_level / noise_level
            else:
                snr = 0
            
            # Umbral adaptativo
            threshold = noise_level + (signal_level - noise_level) * 0.6
            min_distance = int(SAMPLE_RATE * 0.1)  # Mínimo 100ms entre picos
            
            # Detectar picos
            peaks, _ = signal.find_peaks(
                envelope_smooth,
                height=threshold,
                distance=min_distance
            )
            
            # Solo mantener picos recientes (último segundo)
            recent_window = SAMPLE_RATE
            recent_peaks = [p for p in peaks if p > len(buffer) - recent_window]
            
            # Almacenar tiempos de picos detectados
            timestamp = time.time()
            new_peaks = []
            for p in recent_peaks:
                # Convertir índice a tiempo
                peak_time = timestamp - (len(buffer) - p) / SAMPLE_RATE
                new_peaks.append((peak_time, envelope_smooth[p]))
            
            # Actualizar picos detectados
            detected_peaks = new_peaks
            
            # Mantener solo picos recientes (últimos 5 segundos)
            current_time = time.time()
            peak_times = [(t, v) for t, v in peak_times if current_time - t < 5.0]
            peak_times.extend(new_peaks)
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error en procesamiento: {e}")

def update_plot(frame, *fargs):
    """Actualiza la visualización"""
    ax1, ax2, ax3, fig = fargs
    
    try:
        # Obtener datos del buffer
        if q.empty():
            return (ax1, ax2, ax3)
        
        # Limpiar gráficos
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Graficar forma de onda (último segundo)
        try:
            buffer = np.zeros((SAMPLE_RATE,))
            recent_data = []
            
            # Colectar datos recientes
            while not q.empty() and len(recent_data) < SAMPLE_RATE:
                data = q.get_nowait()
                recent_data.extend(data.flatten())
            
            # Mantener solo el último segundo
            if len(recent_data) > SAMPLE_RATE:
                recent_data = recent_data[-SAMPLE_RATE:]
            
            # Actualizar buffer
            buffer[-len(recent_data):] = recent_data
            
            # Visualizar forma de onda
            ax1.set_title("Forma de onda (Entrada directa)")
            ax1.set_ylim(-MAX_AMPLITUDE, MAX_AMPLITUDE)
            time_axis = np.linspace(-1, 0, len(buffer))
            ax1.plot(time_axis, buffer)
            ax1.grid(True)
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel("Amplitud")
            
            # Re-filtrar para visualización
            sos_high = signal.butter(4, 800, 'highpass', fs=SAMPLE_RATE, output='sos')
            sos_band = signal.butter(6, [1000, 8000], 'bandpass', fs=SAMPLE_RATE, output='sos')
            
            filtered = signal.sosfilt(sos_high, buffer)
            filtered = signal.sosfilt(sos_band, filtered)
            
            # Visualizar señal filtrada
            ax2.set_title("Señal filtrada (800Hz-8kHz)")
            ax2.set_ylim(-MAX_AMPLITUDE, MAX_AMPLITUDE)
            ax2.plot(time_axis, filtered)
            ax2.grid(True)
            ax2.set_xlabel("Tiempo (s)")
            ax2.set_ylabel("Amplitud")
            
            # Envolvente y picos
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)
            
            win_length = int(SAMPLE_RATE * 0.001)  # 1ms
            if win_length % 2 == 0:
                win_length += 1
            envelope_smooth = signal.savgol_filter(envelope, win_length, 2)
            
            # Umbral adaptativo
            threshold = noise_level + (signal_level - noise_level) * 0.6
            
            # Visualizar envolvente y umbral
            ax3.set_title(f"Envolvente y detección de picos (SNR: {snr:.2f})")
            ax3.set_ylim(0, MAX_AMPLITUDE)
            ax3.plot(time_axis, envelope_smooth)
            ax3.axhline(y=threshold, color='r', linestyle='--', label=f"Umbral ({threshold:.4f})")
            
            # Marcar picos detectados
            current_time = time.time()
            for t, v in peak_times:
                if current_time - t <= 1.0:  # Solo mostrar picos del último segundo
                    # Convertir tiempo a índice
                    idx = int(len(buffer) - (current_time - t) * SAMPLE_RATE)
                    if 0 <= idx < len(buffer):
                        ax3.plot(time_axis[idx], v, 'ro')
            
            ax3.grid(True)
            ax3.set_xlabel("Tiempo (s)")
            ax3.set_ylabel("Amplitud")
            ax3.legend()
            
            # Información de diagnóstico
            peak_count = len([t for t, _ in peak_times if current_time - t <= 3.0])
            info_text = (
                f"Nivel de señal: {signal_level:.6f}\n"
                f"Nivel de ruido: {noise_level:.6f}\n"
                f"SNR: {snr:.2f}\n"
                f"Picos detectados (3s): {peak_count}"
            )
            
            plt.figtext(0.02, 0.02, info_text, bbox=dict(facecolor='white', alpha=0.8))
            
        except Exception as e:
            print(f"Error en actualización: {e}")
        
        fig.tight_layout()
        return (ax1, ax2, ax3)
        
    except Exception as e:
        print(f"Error en visualización: {e}")
        return (ax1, ax2, ax3)

def main():
    parser = argparse.ArgumentParser(description='Herramienta de diagnóstico de audio para timegrapher')
    parser.add_argument('--device', type=int, help='Índice del dispositivo de entrada')
    parser.add_argument('--gain', type=float, default=10.0, help='Pre-amplificación')
    args = parser.parse_args()
    
    global PRE_GAIN
    PRE_GAIN = args.gain
    
    # Mostrar dispositivos disponibles
    print("Dispositivos de entrada disponibles:")
    print(sd.query_devices())
    
    # Iniciar procesamiento en segundo plano
    processor = threading.Thread(target=process_audio)
    processor.daemon = True
    processor.start()
    
    # Configurar gráfico
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Diagnóstico de Audio para Timegrapher')
    
    # Configurar animación
    ani = FuncAnimation(
        fig, update_plot, interval=100, blit=True, fargs=(ax1, ax2, ax3, fig)
    )
    
    # Iniciar stream de audio
    with sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BUFFER_SIZE,
        callback=audio_callback
    ):
        print(f"Stream de audio iniciado. Pre-ganancia: {PRE_GAIN}x")
        plt.show()

if __name__ == "__main__":
    main()
