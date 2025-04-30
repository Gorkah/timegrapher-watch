# Desktop Timegrapher

A professional desktop timegrapher application for analyzing mechanical watch performance. This application uses a computer's microphone to capture the sound of a watch's escapement and analyze its timing, beat error, and amplitude in real-time.

## Features

- Real-time audio capture from microphone input
- Analysis of watch timing parameters:
  - Rate/daily deviation (s/day)
  - Beat error (ms)
  - Amplitude (degrees)
- Multiple visualization modes:
  - Timing trace graph
  - Beat error display
  - Amplitude measurement
- Support for different watch beat rates (18,000, 21,600, 28,800, 36,000 BPH)
- Recording and exporting of analysis results

## Requirements

- Python 3.8+
- Audio input device (microphone)
- Supported operating systems: Windows, macOS, Linux

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python src/main.py
   ```

## Usage

1. Place a mechanical watch near your microphone
2. Select the appropriate beat rate for your watch
3. Start the analysis
4. View real-time results and adjust microphone position as needed

## License

MIT License
