# ⏱️ Precision Timegrapher

**Analizador de precisión para relojes mecánicos**, usando cualquier micrófono conectado al PC. Ideal para entusiastas y relojeros que desean verificar la frecuencia y el comportamiento de sus relojes en cualquier entorno.

---

## 🚀 Características

- 🔊 Utiliza cualquier micrófono como entrada de audio.
- 📡 Detección automática de la frecuencia del reloj (por ejemplo: 18000, 21600, 28800 bph).
- 🧹 Filtros de ruido y ajuste automático de umbral mínimo para funcionar incluso en ambientes no ideales.
- 📈 Visualización clara de los pulsos y cálculo de la desviación en segundos por día.

---

## ⚙️ Instalación

1. Clona el repositorio o descarga los archivos.

2. Abre una terminal en la carpeta del proyecto:

```bash
cd desktop-timegrapher
```
3. Instala las dependencias necesarias
```bash
pip install -r src/requirements.txt
```

## ▶️ Uso
Ejecuta el analizador desde el directorio raíz:
```bash
py src/precision_timegrapher.py
```

Una vez iniciado, el programa capturará el sonido del reloj automáticamente y mostrará los resultados en tiempo real.

---

## 🔧 Próximas mejoras

- 🎯 Calibración automática de amplitud.  
- 📊 Exportación de resultados a CSV y PDF.  
- 🌐 Interfaz web para control remoto.  
- 📱 Versión móvil con micrófono incorporado.  
- 🔔 Alertas de error en la lectura.  

---

## 📬 Contacto y soporte

¿Tienes ideas, preguntas o quieres colaborar?  
¡Estás invitado a contribuir o abrir un *issue*!

---

## 🛑 Aviso

Este proyecto está en desarrollo activo.  
Su precisión dependerá de la calidad del micrófono y del entorno acústico.  
Se recomienda su uso con micrófonos de contacto o colocación cercana al mecanismo del reloj para mejores resultados.


