"""
Complete Web-based Real-time STT Streaming Server with GPU Support
Optimized for RTX 4090 with WebSocket support for multiple clients
"""

import asyncio
import json
import logging
import time
import base64
from pathlib import Path
from typing import Dict, Optional, List
import uuid
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUSTTStreaming:
    """
    GPU-optimized STT streaming class for multiple concurrent connections
    """
    
    
    
    def __init__(self, 
                 model_name: str = "kyutai/stt-1b-en_fr",  # Use 1B model for better stability
                 device: str = "cuda",
                 batch_size: int = 1,  # Start with batch size 1 to avoid tensor issues
                 sample_rate: int = 16000):
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        
        # Model components
        self.mimi = None
        self.moshi = None
        self.lm_gen = None
        self.text_tokenizer = None
        self.frame_size = None
        self.is_initialized = False
        
        # Client management
        self.active_clients: Dict[str, dict] = {}
        self.client_buffers: Dict[str, np.ndarray] = {}
        self.client_states: Dict[str, dict] = {}  # Per-client model states
        
    async def initialize_model(self):
        """Initialize the Moshi STT model with GPU optimization"""
        try:
            from moshi.models import LMGen, loaders
            
            logger.info(f"🚀 Loading Kyutai STT model: {self.model_name}")
            logger.info(f"🎮 Using device: {self.device}")
            logger.info(f"📊 Batch size: {self.batch_size}")
            
            # Check GPU availability
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.batch_size = 1
            
            # Load model configuration
            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(self.model_name)
            
            # Initialize audio encoder (Mimi) with GPU
            self.mimi = checkpoint_info.get_mimi(device=self.device)
            self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
            
            logger.info(f"📡 Model sample rate: {self.mimi.sample_rate}")
            logger.info(f"🔢 Frame size: {self.frame_size}")
            
            # Initialize language model (Moshi)
            self.moshi = checkpoint_info.get_moshi(device=self.device)
            self.lm_gen = LMGen(self.moshi, temp=0, temp_text=0)
            
            # Setup streaming mode with batch processing
            self.mimi.streaming_forever(self.batch_size)
            self.lm_gen.streaming_forever(self.batch_size)
            
            # Get text tokenizer
            self.text_tokenizer = checkpoint_info.get_text_tokenizer()
            
            # GPU warmup
            await self._warmup_gpu()
            
            self.is_initialized = True
            logger.info("✅ Model initialized successfully on GPU!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing model: {e}")
            raise
    
    async def _warmup_gpu(self):
        """Warm up GPU with dummy data"""
        logger.info("🔥 Warming up GPU...")
        try:
            dummy_audio = torch.zeros(self.batch_size, 1, self.frame_size, device=self.device)
            codes = self.mimi.encode(dummy_audio)
            
            for c in range(min(10, codes.shape[-1])):  # Limited warmup
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is not None:
                    break
            
            # Synchronize GPU
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            logger.info("✅ GPU warmed up")
        except Exception as e:
            logger.warning(f"⚠️ GPU warmup failed: {e}")
    
    def add_client(self, client_id: str):
        """Add a new streaming client"""
        self.active_clients[client_id] = {
            'connected_at': time.time(),
            'processed_chunks': 0,
            'batch_slot': None
        }
        self.client_buffers[client_id] = np.array([], dtype=np.float32)
        logger.info(f"👤 Client {client_id[:8]}... connected ({len(self.active_clients)} total)")
    
    def remove_client(self, client_id: str):
        """Remove a streaming client"""
        if client_id in self.active_clients:
            del self.active_clients[client_id]
        if client_id in self.client_buffers:
            del self.client_buffers[client_id]
        if client_id in self.client_states:
            del self.client_states[client_id]
        logger.info(f"👋 Client {client_id[:8]}... disconnected ({len(self.active_clients)} total)")
        # Force garbage collection to free memory
        import gc
        gc.collect()
    
    def reset_client_state(self, client_id: str):
        """Reset streaming state for a specific client"""
        if client_id in self.client_buffers:
            self.client_buffers[client_id] = np.array([], dtype=np.float32)
        
        if client_id in self.client_states:
            del self.client_states[client_id]
        
        # For now, reset global model state
        # TODO: Implement per-client model states for true multi-client support
        if self.is_initialized:
            try:
                self.mimi.reset_streaming()
                self.lm_gen.reset_streaming()
                logger.info(f"🔄 Reset state for client {client_id[:8]}...")
            except Exception as e:
                logger.warning(f"⚠️ State reset error: {e}")
    
    async def process_audio_data(self, client_id: str, audio_data: bytes) -> List[str]:
        """
        Process audio data from a specific client
        
        Args:
            client_id: Unique client identifier
            audio_data: Raw audio bytes
            
        Returns:
            List of transcribed text pieces
        """
        if not self.is_initialized or client_id not in self.active_clients:
            return []
        
        try:
            # Ensure audio_data length is even (for 16-bit samples)
            if len(audio_data) % 2 != 0:
                audio_data = audio_data[:-1]  # Remove last byte if odd
            
            if len(audio_data) == 0:
                return []
            
            # Convert audio bytes to numpy array (16-bit PCM)
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            except ValueError as e:
                logger.error(f"❌ Audio conversion error for client {client_id[:8]}: {e}")
                return []
            
            if len(audio_array) == 0:
                return []
            
            # Resample if necessary (Kyutai models expect 24kHz, but we receive 16kHz)
            if self.mimi.sample_rate != 16000:
                # Simple upsampling by linear interpolation
                upsample_factor = self.mimi.sample_rate / 16000
                if upsample_factor != 1.0:
                    # Use numpy interpolation for better quality
                    original_length = len(audio_array)
                    new_length = int(original_length * upsample_factor)
                    audio_array = np.interp(
                        np.linspace(0, original_length-1, new_length),
                        np.arange(original_length),
                        audio_array
                    ).astype(np.float32)
            
            # Add to client buffer
            self.client_buffers[client_id] = np.concatenate([
                self.client_buffers[client_id], 
                audio_array
            ])
            
            results = []
            buffer = self.client_buffers[client_id]
            
            # Process complete frames
            while len(buffer) >= self.frame_size:
                frame = buffer[:self.frame_size]
                buffer = buffer[self.frame_size:]
                
                # Process frame
                frame_results = await self._process_audio_frame(frame, client_id)
                results.extend(frame_results)
            
            # Update buffer
            self.client_buffers[client_id] = buffer
            
            # Update client stats
            self.active_clients[client_id]['processed_chunks'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing audio for client {client_id[:8]}: {e}")
            return []
    
    async def _process_audio_frame(self, frame: np.ndarray, client_id: str) -> List[str]:
        """Process a single audio frame through the model"""
        try:
            # Ensure frame is the correct size
            if len(frame) != self.frame_size:
                return []
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(frame.astype(np.float32))
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, frame_size)
            audio_tensor = audio_tensor.to(self.device)
            
            results = []
            
            with torch.no_grad():
                try:
                    # Encode audio - use single batch for now to avoid tensor size issues
                    codes = self.mimi.encode(audio_tensor)
                    
                    if codes is None:
                        return []
                    
                    # Process each code
                    for c in range(codes.shape[-1]):
                        try:
                            code_slice = codes[:, :, c : c + 1]
                            result = self.lm_gen.step_with_extra_heads(code_slice)
                            
                            if result is None:
                                continue
                            
                            text_tokens, vad_heads = result
                            
                            if text_tokens is None:
                                continue
                            
                            # Check Voice Activity Detection
                            if vad_heads and len(vad_heads) > 2:
                                try:
                                    vad_prob = vad_heads[2][0, 0, 0].cpu().item()
                                    if vad_prob > 0.5:
                                        results.append("[END_TURN]")
                                        continue
                                except (IndexError, RuntimeError):
                                    pass  # Skip VAD if there's an error
                            
                            # Extract text token
                            try:
                                text_token = text_tokens[0, 0, 0].item()
                                
                                if text_token not in (0, 3):  # Skip special tokens
                                    text_piece = self.text_tokenizer.id_to_piece(text_token)
                                    text_piece = text_piece.replace("▁", " ")
                                    
                                    if text_piece.strip():
                                        results.append(text_piece)
                            except (IndexError, RuntimeError):
                                continue  # Skip this token if there's an error
                            
                        except Exception as e:
                            logger.debug(f"Code processing error: {e}")
                            continue  # Skip this code and continue
                
                except Exception as e:
                    logger.error(f"❌ Encoding error for client {client_id[:8]}: {e}")
                    return []
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")
            return []



# modelList = ["kyutai/stt-1b-en_fr", "kyutai/stt-2.6b-en"]
# Global STT instance
stt_engine = GPUSTTStreaming()

# FastAPI application with lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await stt_engine.initialize_model()
    yield
    # Shutdown - cleanup if needed
    logger.info("🛑 Server shutting down")

app = FastAPI(
    title="Kyutai STT Streaming Server", 
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def get_demo_page():
    """Serve the demo HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kyutai STT Streaming Demo</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #020202 0%, #161616 100%);
                color: white;
                min-height: 100vh;
                line-height: 1;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: #2d2d2d;
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
            }
            h1{
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            h2{
                text-align: center;
                margin-bottom: 30px;
                font-size: 1.8rem;
            }
            a{
                color: lightgreen;
            }
            .controls {
                display: grid;
                justify-content: center;
                gap: 20px;
                margin-bottom: 30px;
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none !important;
            }
            .start-btn {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
            }
            .start-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .stop-btn {
                background: linear-gradient(45deg, #f44336, #da190b);
                color: white;
            }
            .stop-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .clear-btn {
                background: linear-gradient(45deg, #ff9800, #f57c00);
                color: white;
            }
            .clear-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .permission-btn {
                background: linear-gradient(45deg, #ffc107, #ffb300);
                color: #000;
                font-weight: bold;
                margin: 10px 0;
            }
            .permission-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .status {
                text-align: center;
                margin-bottom: 20px;
                font-size: 18px;
                font-weight: bold;
            }
            .status.recording {
                color: #4CAF50;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .permission-check {
                text-align: center;
                margin: 20px 0;
                padding: 20px;
                background: rgba(255, 193, 7, 0.1);
                border-radius: 10px;
                border-left: 4px solid #ffc107;
            }
            .transcription {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
                padding: 20px;
                min-height: 200px;
                font-size: 16px;
                line-height: 1.6;
                word-wrap: break-word;
                border-left: 4px solid #4CAF50;
            }
            .partial-text {
                color: #ffeb3b;
                font-style: italic;
            }
            .final-text {
                color: white;
                margin-bottom: 10px;
                padding: 5px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 5px;
            }
            .stats, .credits {
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
                font-size: 14px;
                opacity: 0.8;
            }
            .error {
                color: #ff5722;
                background: rgba(255, 87, 34, 0.1);
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                border-left: 4px solid #ff5722;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div>
                <h1>Speech-to-Text Streaming</h1>
                <h2>Powered by: <a href="https://kyutai.org/" target="_blank">Kyutai-stt</a></h2>
            </div>
            
            <div class="controls">
                <button id="permissionBtn" class="permission-btn">Check Mic Permission</button>
                <button id="clearBtn" class="clear-btn">Clear Text</button>
                
                <div>
                    <button id="startBtn" class="start-btn">Start Recording</button>
                    <button id="stopBtn" class="stop-btn" disabled>Stop Recording</button>
                </div>
            </div>
            
            <div id="permissionStatus" class="permission-check" style="display: none;">
                <div>Microphone permission is required for speech recognition</div>
                <div style="font-size: 14px; margin-top: 10px; opacity: 0.8;">
                    Click "Check Microphone Permission" first, then allow access when prompted by your browser.
                </div>
            </div>
            
            <div id="status" class="status">Ready to record</div>
            
            <div id="transcription" class="transcription">
                <div style="text-align: center; opacity: 0.6; font-style: italic;">
                    Transcribed text will appear here...
                </div>
            </div>
            
            <div class="stats">
                <div>Connection: <span id="connectionStatus">Disconnected</span></div>
                <div>Messages sent: <span id="messageCount">0</span></div>
                <div>Recording time: <span id="recordingTime">00:00</span></div>
            </div>
            <div class="credits">
                <div>Author: Soh-Kaz</div>
                <div>Github: <a href="https://github.com/soh-kaz">Soh-Kaz</a></div>
            </div>
        </div>

        <script>
            class STTStreaming {
                constructor() {
                    this.ws = null;
                    this.audioContext = null;
                    this.mediaStream = null;
                    this.processor = null;
                    this.workletNode = null;
                    this.isRecording = false;
                    this.messageCount = 0;
                    this.startTime = null;
                    this.timerInterval = null;
                    this.currentPartialText = '';
                    this.retryCount = 0;
                    this.maxRetries = 5;
                    
                    this.initializeElements();
                    this.connectWebSocket();
                }
                
                initializeElements() {
                    this.permissionBtn = document.getElementById('permissionBtn');
                    this.startBtn = document.getElementById('startBtn');
                    this.stopBtn = document.getElementById('stopBtn');
                    this.clearBtn = document.getElementById('clearBtn');
                    this.status = document.getElementById('status');
                    this.transcription = document.getElementById('transcription');
                    this.connectionStatus = document.getElementById('connectionStatus');
                    this.messageCountEl = document.getElementById('messageCount');
                    this.recordingTimeEl = document.getElementById('recordingTime');
                    this.permissionStatus = document.getElementById('permissionStatus');
                    
                    this.permissionBtn.addEventListener('click', () => this.checkMicrophonePermission());
                    this.startBtn.addEventListener('click', () => this.startRecording());
                    this.stopBtn.addEventListener('click', () => this.stopRecording());
                    this.clearBtn.addEventListener('click', () => this.clearTranscription());
                    
                    // Check initial permission status
                    this.checkInitialPermissions();
                }
                
                async checkInitialPermissions() {
                    // Check if we're on HTTPS or localhost
                    const isSecureContext = window.isSecureContext || 
                                          location.protocol === 'https:' || 
                                          location.hostname === 'localhost' || 
                                          location.hostname === '127.0.0.1';
                    
                    if (!isSecureContext) {
                        this.showHttpsWarning();
                        return;
                    }
                    
                    // Check if getUserMedia is available
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        this.showBrowserCompatibilityError();
                        return;
                    }
                    
                    try {
                        if (navigator.permissions && navigator.permissions.query) {
                            const permission = await navigator.permissions.query({ name: 'microphone' });
                            this.handlePermissionStatus(permission.state);
                            
                            permission.onchange = () => {
                                this.handlePermissionStatus(permission.state);
                            };
                        } else {
                            // Fallback for browsers that don't support permissions API
                            this.permissionStatus.style.display = 'block';
                        }
                    } catch (error) {
                        console.warn('Could not check microphone permissions:', error);
                        this.permissionStatus.style.display = 'block';
                    }
                }
                
                showHttpsWarning() {
                    this.permissionStatus.innerHTML = `
                        <div style="color: #ff5722;">
                            <h3>🔒 HTTPS Required for Microphone Access</h3>
                            <p>Modern browsers require HTTPS for microphone access for security reasons.</p>
                            <div style="margin: 15px 0; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;">
                                <strong>Solutions:</strong><br>
                                1. <strong>Use HTTPS:</strong> Access via https://${location.host}<br>
                                2. <strong>Use localhost:</strong> Access via http://localhost:8000 (if running locally)<br>
                                3. <strong>Self-signed certificate:</strong> Enable HTTPS on the server
                            </div>
                            <button onclick="window.location.href='https://${location.host}'" 
                                    style="padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px;">
                                🔒 Try HTTPS
                            </button>
                            <button onclick="window.location.href='http://localhost:8000'" 
                                    style="padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px;">
                                🏠 Try Localhost
                            </button>
                        </div>
                    `;
                    this.permissionStatus.style.display = 'block';
                    this.startBtn.disabled = true;
                    this.permissionBtn.disabled = true;
                }
                
                showBrowserCompatibilityError() {
                    this.permissionStatus.innerHTML = `
                        <div style="color: #ff5722;">
                            <h3>❌ Browser Not Supported</h3>
                            <p>Your browser doesn't support the required audio features.</p>
                            <div style="margin: 15px 0;">
                                <strong>Supported browsers:</strong><br>
                                • Chrome 66+<br>
                                • Firefox 55+<br>
                                • Safari 11+<br>
                                • Edge 79+
                            </div>
                            <p>Please update your browser or try a different one.</p>
                        </div>
                    `;
                    this.permissionStatus.style.display = 'block';
                    this.startBtn.disabled = true;
                    this.permissionBtn.disabled = true;
                }
                
                handlePermissionStatus(state) {
                    if (state === 'granted') {
                        this.permissionStatus.style.display = 'none';
                        this.startBtn.disabled = false;
                        this.permissionBtn.textContent = 'Permission Granted';
                        this.permissionBtn.style.background = 'linear-gradient(45deg, #4CAF50, #45a049)';
                        this.permissionBtn.style.color = 'white';
                    } else if (state === 'denied') {
                        this.permissionStatus.style.display = 'block';
                        this.startBtn.disabled = true;
                        this.permissionBtn.textContent = 'Permission Denied';
                        this.permissionBtn.style.background = 'linear-gradient(45deg, #f44336, #da190b)';
                        this.permissionBtn.style.color = 'white';
                        this.showError('Microphone access denied. Please enable it in your browser settings.');
                    } else {
                        this.permissionStatus.style.display = 'block';
                        this.startBtn.disabled = true;
                        this.permissionBtn.textContent = '🔐 Check Microphone Permission';
                    }
                }
                
                async checkMicrophonePermission() {
                    // Check HTTPS requirement first
                    const isSecureContext = window.isSecureContext || 
                                          location.protocol === 'https:' || 
                                          location.hostname === 'localhost' || 
                                          location.hostname === '127.0.0.1';
                    
                    if (!isSecureContext) {
                        this.showHttpsWarning();
                        return;
                    }
                    
                    // Check browser compatibility
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        this.showBrowserCompatibilityError();
                        return;
                    }
                    
                    try {
                        this.permissionBtn.textContent = 'Checking permissions...';
                        this.permissionBtn.disabled = true;
                        
                        // Request microphone access to check permissions
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            audio: true 
                        });
                        
                        // If successful, stop the stream immediately
                        stream.getTracks().forEach(track => track.stop());
                        
                        this.handlePermissionStatus('granted');
                        this.showStatus('Microphone permission granted! You can now start recording.');
                        
                    } catch (error) {
                        console.error('Permission check failed:', error);
                        
                        if (error.name === 'NotAllowedError') {
                            this.handlePermissionStatus('denied');
                        } else if (error.name === 'NotFoundError') {
                            this.showError('No microphone found. Please connect a microphone and try again.');
                            this.permissionBtn.textContent = 'No Microphone Found';
                        } else {
                            this.showError(`Permission check failed: ${error.message}`);
                            this.permissionBtn.textContent = 'Permission Check Failed';
                        }
                    } finally {
                        this.permissionBtn.disabled = false;
                    }
                }
                
                connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    const attemptConnection = () => {
                        if (this.retryCount >= this.maxRetries) {
                            this.showError('Failed to connect to WebSocket server after multiple attempts.');
                            return;
                        }

                        this.ws = new WebSocket(wsUrl);
                        
                        this.ws.onopen = () => {
                            this.connectionStatus.textContent = 'Connected';
                            this.connectionStatus.style.color = '#4CAF50';
                            console.log('WebSocket connected');
                            this.retryCount = 0; // Reset retry count on success
                        };
                        
                        this.ws.onclose = () => {
                            this.connectionStatus.textContent = 'Disconnected';
                            this.connectionStatus.style.color = '#f44336';
                            console.log('WebSocket disconnected');
                            this.retryCount++;
                            setTimeout(attemptConnection, 3000);
                        };
                        
                        this.ws.onmessage = (event) => {
                            try {
                                const data = JSON.parse(event.data);
                                this.handleTranscriptionResult(data);
                            } catch (error) {
                                console.error('Error parsing message:', error);
                                this.showError('Error processing server response');
                            }
                        };
                        
                        this.ws.onerror = (error) => {
                            console.error('WebSocket error:', error);
                            this.showError('WebSocket connection error');
                        };
                    };
                    
                    attemptConnection();
                }
                
                async startRecording() {
                    try {
                        // Disable start button immediately to prevent multiple clicks
                        this.startBtn.disabled = true;
                        console.log('Requesting microphone access...');
                        this.showStatus('Requesting microphone permission...');
                        
                        // Check if getUserMedia is supported
                        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                            throw new Error('getUserMedia is not supported in this browser');
                        }
                        
                        // Request microphone access with constraints
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            audio: {
                                sampleRate: 16000,  // Request 16kHz
                                channelCount: 1,
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true
                            }
                        });
                        
                        console.log('Microphone access granted');
                        this.showStatus('Microphone access granted, starting recording...');
                        
                        // Get the actual sample rate from the stream
                        const audioTrack = stream.getAudioTracks()[0];
                        const settings = audioTrack.getSettings();
                        const inputSampleRate = settings.sampleRate || 48000; // Fallback to 48kHz if not available
                        
                        console.log(`Microphone stream sample rate: ${inputSampleRate}`);
                        
                        // Create AudioContext with the matching sample rate
                        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                            sampleRate: inputSampleRate
                        });
                        
                        console.log(`AudioContext initialized with sample rate: ${this.audioContext.sampleRate}`);
                        
                        const source = this.audioContext.createMediaStreamSource(stream);
                        
                        // Create a ScriptProcessorNode or AudioWorklet for real-time processing
                        try {
                            if (this.audioContext.audioWorklet) {
                                await this.setupAudioWorklet(source);
                            } else {
                                this.setupScriptProcessor(source);
                            }
                        } catch (error) {
                            console.error('Error setting up audio processing:', error);
                            throw new Error(`Audio processing setup failed: ${error.message}`);
                        }
                        
                        this.mediaStream = stream;
                        this.isRecording = true;
                        this.startTime = Date.now();
                        
                        this.updateUI();
                        this.startTimer();
                        this.showStatus('🔴 Recording... Speak into your microphone');
                        
                    } catch (error) {
                        console.error('Error starting recording:', error);
                        let errorMessage = 'Could not access microphone. ';
                        
                        if (error.name === 'NotAllowedError') {
                            errorMessage += 'Permission denied. Please allow microphone access and try again.';
                        } else if (error.name === 'NotFoundError') {
                            errorMessage += 'No microphone found. Please connect a microphone and try again.';
                        } else if (error.name === 'NotReadableError') {
                            errorMessage += 'Microphone is being used by another application.';
                        } else {
                            errorMessage += error.message;
                        }
                        
                        this.showError(errorMessage);
                        this.showStatus('❌ Recording failed');
                        this.updateUI(); // Ensure UI is updated to reflect failure
                    }
                }
                
                async resampleAudioBuffer(audioBuffer, sourceSampleRate, targetSampleRate) {
                    try {
                        const offlineCtx = new OfflineAudioContext(1, audioBuffer.length, sourceSampleRate);
                        const buffer = offlineCtx.createBuffer(1, audioBuffer.length, sourceSampleRate);
                        buffer.getChannelData(0).set(audioBuffer);
                        
                        const source = offlineCtx.createBufferSource();
                        source.buffer = buffer;
                        source.connect(offlineCtx.destination);
                        source.start();
                        
                        const renderedBuffer = await offlineCtx.startRendering();
                        const originalLength = renderedBuffer.length;
                        const targetLength = Math.round(originalLength * (targetSampleRate / sourceSampleRate));
                        
                        const targetCtx = new OfflineAudioContext(1, targetLength, targetSampleRate);
                        const targetBuffer = targetCtx.createBuffer(1, targetLength, targetSampleRate);
                        const targetData = targetBuffer.getChannelData(0);
                        
                        // Linear interpolation for resampling
                        const originalData = renderedBuffer.getChannelData(0);
                        for (let i = 0; i < targetLength; i++) {
                            const index = i * (originalLength / targetLength);
                            const lowIndex = Math.floor(index);
                            const highIndex = Math.ceil(index);
                            const frac = index - lowIndex;
                            
                            if (highIndex >= originalLength) {
                                targetData[i] = originalData[originalLength - 1];
                            } else {
                                targetData[i] = originalData[lowIndex] * (1 - frac) + originalData[highIndex] * frac;
                            }
                        }
                        
                        return targetData;
                    } catch (error) {
                        console.error('Resampling error:', error);
                        throw error;
                    }
                }
                
                setupScriptProcessor(source) {
                    // Fallback method using ScriptProcessorNode
                    const bufferSize = 4096;
                    this.processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
                    
                    this.processor.onaudioprocess = (event) => {
                        if (this.isRecording) {
                            const inputBuffer = event.inputBuffer.getChannelData(0);
                            this.sendAudioBuffer(inputBuffer);
                        }
                    };
                    
                    source.connect(this.processor);
                    this.processor.connect(this.audioContext.destination);
                }
                
                async setupAudioWorklet(source) {
                    // Modern approach with AudioWorklet
                    const workletCode = `
                        class AudioProcessor extends AudioWorkletProcessor {
                            process(inputs, outputs, parameters) {
                                const input = inputs[0];
                                if (input && input[0]) {
                                    this.port.postMessage({
                                        type: 'audiodata',
                                        data: input[0]
                                    });
                                }
                                return true;
                            }
                        }
                        registerProcessor('audio-processor', AudioProcessor);
                    `;
                    
                    const blob = new Blob([workletCode], { type: 'application/javascript' });
                    const workletUrl = URL.createObjectURL(blob);
                    
                    await this.audioContext.audioWorklet.addModule(workletUrl);
                    
                    this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');
                    this.workletNode.port.onmessage = (event) => {
                        if (event.data.type === 'audiodata' && this.isRecording) {
                            this.sendAudioBuffer(event.data.data);
                        }
                    };
                    
                    source.connect(this.workletNode);
                    this.workletNode.connect(this.audioContext.destination);
                }
                
                stopRecording() {
                    if (this.isRecording) {
                        this.isRecording = false;
                        
                        // Stop media stream
                        if (this.mediaStream) {
                            this.mediaStream.getTracks().forEach(track => track.stop());
                            this.mediaStream = null;
                        }
                        
                        // Clean up audio processing
                        if (this.processor) {
                            this.processor.disconnect();
                            this.processor = null;
                        }
                        
                        if (this.workletNode) {
                            this.workletNode.disconnect();
                            this.workletNode = null;
                        }
                        
                        if (this.audioContext && this.audioContext.state !== 'closed') {
                            this.audioContext.close();
                            this.audioContext = null;
                        }
                        
                        this.updateUI();
                        this.stopTimer();
                        this.showStatus('Recording stopped');
                        console.log('Recording stopped');
                    }
                }
                
                async sendAudioBuffer(audioBuffer) {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN && audioBuffer.length > 0) {
                        try {
                            // Resample to 16kHz if needed
                            const targetSampleRate = 16000;
                            let processedBuffer = audioBuffer;
                            if (this.audioContext.sampleRate !== targetSampleRate) {
                                console.log(`Resampling from ${this.audioContext.sampleRate}Hz to ${targetSampleRate}Hz`);
                                processedBuffer = await this.resampleAudioBuffer(
                                    audioBuffer,
                                    this.audioContext.sampleRate,
                                    targetSampleRate
                                );
                            }
                            
                            // Convert Float32Array to Int16Array for better compression
                            const int16Array = new Int16Array(processedBuffer.length);
                            for (let i = 0; i < processedBuffer.length; i++) {
                                int16Array[i] = Math.max(-32768, Math.min(32767, processedBuffer[i] * 32768));
                            }
                            
                            // Convert to base64
                            const bytes = new Uint8Array(int16Array.buffer);
                            const base64 = btoa(String.fromCharCode(...bytes));
                            
                            this.ws.send(JSON.stringify({
                                type: 'audio',
                                data: base64,
                                sampleRate: targetSampleRate
                            }));
                            
                            this.messageCount++;
                            this.messageCountEl.textContent = this.messageCount;
                            
                        } catch (error) {
                            console.error('Error sending audio buffer:', error);
                            this.showError(`Failed to process audio: ${error.message}`);
                        }
                    }
                }
                
                handleTranscriptionResult(data) {
                    if (data.type === 'transcription') {
                        if (data.is_final) {
                            // Final text
                            const finalDiv = document.createElement('div');
                            finalDiv.className = 'final-text';
                            finalDiv.textContent = data.text;
                            this.transcription.appendChild(finalDiv);
                            this.currentPartialText = '';
                        } else {
                            // Partial text
                            this.currentPartialText += data.text;
                            this.updatePartialText();
                        }
                        
                        // Auto-scroll
                        this.transcription.scrollTop = this.transcription.scrollHeight;
                    } else if (data.type === 'error') {
                        this.showError(data.message);
                    }
                }
                
                updatePartialText() {
                    let partialDiv = document.querySelector('.partial-text');
                    if (!partialDiv) {
                        partialDiv = document.createElement('div');
                        partialDiv.className = 'partial-text';
                        this.transcription.appendChild(partialDiv);
                    }
                    partialDiv.textContent = this.currentPartialText;
                }
                
                clearTranscription() {
                    this.transcription.innerHTML = '<div style="text-align: center; opacity: 0.6; font-style: italic;">Transcribed text will appear here...</div>';
                    this.currentPartialText = '';
                }
                
                showStatus(message) {
                    this.status.textContent = message;
                }
                
                updateUI() {
                    if (this.isRecording) {
                        this.startBtn.disabled = true;
                        this.stopBtn.disabled = false;
                        this.status.className = 'status recording';
                    } else {
                        this.startBtn.disabled = false;
                        this.stopBtn.disabled = true;
                        this.status.className = 'status';
                    }
                }
                
                startTimer() {
                    this.timerInterval = setInterval(() => {
                        if (this.startTime) {
                            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
                            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
                            const seconds = (elapsed % 60).toString().padStart(2, '0');
                            this.recordingTimeEl.textContent = `${minutes}:${seconds}`;
                        }
                    }, 1000);
                }
                
                stopTimer() {
                    if (this.timerInterval) {
                        clearInterval(this.timerInterval);
                        this.timerInterval = null;
                    }
                }
                
                showError(message) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'error';
                    errorDiv.textContent = `Error: ${message}`;
                    this.transcription.appendChild(errorDiv);
                    
                    setTimeout(() => {
                        if (errorDiv.parentNode) {
                            errorDiv.parentNode.removeChild(errorDiv);
                        }
                    }, 5000);
                }
            }
            
            // Initialize the app when page loads
            document.addEventListener('DOMContentLoaded', () => {
                new STTStreaming();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming STT"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    stt_engine.add_client(client_id)
    
    try:
        logger.info(f"🌐 WebSocket client {client_id[:8]}... connected")
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio":
                # Decode base64 audio data
                audio_data = base64.b64decode(message["data"])
                
                # Process audio through STT
                results = await stt_engine.process_audio_data(client_id, audio_data)
                
                # Send results back to client
                for result in results:
                    if result == "[END_TURN]":
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": "",
                            "is_final": True
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": result,
                            "is_final": False
                        }))
            
            elif message.get("type") == "reset":
                # Reset client state
                stt_engine.reset_client_state(client_id)
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "State reset"
                }))
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client {client_id[:8]}... disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error for client {client_id[:8]}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
    finally:
        stt_engine.remove_client(client_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            "gpu_memory_reserved": torch.cuda.memory_reserved(0),
        }
    
    return {
        "status": "healthy",
        "model_initialized": stt_engine.is_initialized,
        "active_clients": len(stt_engine.active_clients),
        "device": stt_engine.device,
        "model_name": stt_engine.model_name,
        "gpu_info": gpu_info
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    client_stats = []
    for client_id, info in stt_engine.active_clients.items():
        client_stats.append({
            "id": client_id[:8] + "...",
            "connected_duration": time.time() - info['connected_at'],
            "processed_chunks": info['processed_chunks']
        })
    
    return {
        "total_clients": len(stt_engine.active_clients),
        "clients": client_stats,
        "model_info": {
            "name": stt_engine.model_name,
            "device": stt_engine.device,
            "batch_size": stt_engine.batch_size,
            "initialized": stt_engine.is_initialized
        }
    }

if __name__ == "__main__":
    import sys
    import os
    
    print("🚀 Starting Kyutai STT Streaming Server")
    print("🎮 GPU-optimized for RTX 4090")
    
    # Check for HTTPS arguments
    use_https = "--https" in sys.argv or "--ssl" in sys.argv
    host = "0.0.0.0"
    port = 8000
    
    # Parse command line arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    if use_https:
        # Check for SSL certificate files
        cert_file = "cert.pem"
        key_file = "key.pem"
        
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            print("❌ SSL certificate files not found!")
            print("📋 To generate self-signed certificates, run:")
            print("   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes")
            print("   or use: python generate_cert.py")
            sys.exit(1)
        
        print(f"🔒 Starting HTTPS server at https://{host}:{port}")
        print("🌐 Web interface will be available at:")
        print(f"   https://localhost:{port} (if running locally)")
        print(f"   https://{host}:{port}")
        print("⚠️  You may need to accept the self-signed certificate in your browser")
        
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level="info"
        )
    else:
        print(f"🌐 Starting HTTP server at http://{host}:{port}")
        print("🌐 Web interface will be available at:")
        print(f"   http://localhost:{port} (microphone will work)")
        print(f"   http://{host}:{port} (microphone may not work - HTTPS recommended)")
        print("")
        print("💡 For microphone access over network, use HTTPS:")
        print(f"   python {sys.argv[0]} --https")
        
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            log_level="info"
        )