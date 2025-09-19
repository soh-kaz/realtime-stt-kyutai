# Real-time Speech-to-Text Streaming Server

A high-performance, GPU-optimized real-time speech-to-text (STT) streaming server built with WebSocket support for multiple concurrent clients. This project leverages the Kyutai STT model and is optimized for NVIDIA RTX 4090 GPUs, providing low-latency transcription for audio streams.

## Features

- **Real-time Transcription**: Streams audio from clients and provides near-instantaneous speech-to-text conversion.
- **GPU Optimization**: Utilizes CUDA for accelerated processing on NVIDIA GPUs (optimized for RTX 4090).
- **Multi-client Support**: Handles multiple WebSocket connections concurrently.
- **Web Interface**: Includes a user-friendly web interface for testing and demo purposes.
- **Error Handling**: Robust error handling for audio processing, WebSocket connections, and microphone permissions.
- **HTTPS Support**: Supports secure connections for microphone access over networks.
- **Health Monitoring**: Provides endpoints for server health and statistics.

## Tech Stack

- **Backend**:
  - Python 3.8+
  - FastAPI: High-performance web framework for building APIs and WebSocket endpoints
  - Uvicorn: ASGI server for running the FastAPI application
  - PyTorch: Deep learning framework for GPU-accelerated STT model processing
  - NumPy: For efficient audio data manipulation
  - Kyutai STT Model: Pre-trained speech-to-text model for transcription
  - WebSocket: For real-time bidirectional communication with clients

- **Frontend**:
  - HTML5/CSS3/JavaScript: For the web-based demo interface
  - Web Audio API: For capturing and processing microphone input
  - AudioWorklet/ScriptProcessorNode: For real-time audio processing in the browser
  - WebSocket API: For client-server communication

- **Infrastructure**:
  - NVIDIA CUDA: For GPU acceleration
  - GitHub: Version control and project hosting
  - HTTPS: Optional SSL/TLS support for secure connections

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optimized for RTX 4090)
- Git
- pip for installing Python dependencies
- OpenSSL (for generating SSL certificates on Ubuntu)
- (Optional) SSL certificates for HTTPS deployment

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/soh-kaz/realtime-stt-kyutai.git
   cd realtime-stt-kyutai
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   fastapi
   uvicorn
   torch
   numpy
   ```

4. **(Optional) Generate SSL Certificates for HTTPS on Ubuntu**:
   Ensure OpenSSL is installed:
   ```bash
   sudo apt update
   sudo apt install openssl
   ```
   Generate self-signed certificates:
   ```bash
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   ```

## Usage

1. **Run the Server (HTTP)**:
   ```bash
   python main.py
   ```
   The server will start at `http://localhost:8000`. Open this URL in a browser to access the web interface.

2. **Run the Server (HTTPS)**:
   ```bash
   python main.py --https
   ```
   Ensure `cert.pem` and `key.pem` are in the project directory. The server will start at `https://localhost:8000`.

3. **Access the Web Interface**:
   - Open `http://localhost:8000` (or `https://localhost:8000` for HTTPS) in a modern browser.
   - Grant microphone permissions when prompted.
   - Click "Start Recording" to begin streaming audio for transcription.
   - View real-time transcriptions in the interface.

4. **API Endpoints**:
   - **GET /**: Serves the demo web interface
   - **GET /health**: Returns server health status and GPU information
   - **GET /stats**: Returns server statistics, including active clients and model info
   - **WebSocket /ws**: WebSocket endpoint for streaming audio and receiving transcriptions

## Web Interface

The web interface provides:
- Microphone permission management
- Start/Stop recording controls
- Real-time transcription display (partial and final text)
- Connection status and recording statistics
- Error notifications
- Clear transcription option

## Notes

- **Microphone Access**: Requires HTTPS for non-localhost access due to browser security restrictions. Use the `--https` flag for network access.
- **Performance**: Optimized for NVIDIA RTX 4090. Adjust `batch_size` in the code for other GPUs to balance performance and memory usage.
- **Model**: Uses Kyutai's `stt-1b-en_fr` model by default. Modify `model_name` in the code to use other compatible models (e.g., `stt-2.6b-en`).
- **Browser Compatibility**: Supports modern browsers (Chrome 66+, Firefox 55+, Safari 11+, Edge 79+).

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## Screenshot
<img src="https://github.com/soh-kaz/realtime-stt-kyutai/blob/main/screenshot.png" />

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- GitHub: [soh-kaz](https://github.com/soh-kaz)
- Repository: [realtime-stt-kyutai](https://github.com/soh-kaz/realtime-stt-kyutai)
