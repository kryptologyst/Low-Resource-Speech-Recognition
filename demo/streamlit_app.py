"""Streamlit demo for low-resource speech recognition."""

import logging
import sys
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Wav2Vec2ASR, ConformerASR
from src.utils import setup_logging, DeviceManager

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Low-Resource Speech Recognition Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Privacy disclaimer
st.markdown("""
<div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336;">
    <h4 style="color: #c62828; margin-top: 0;">PRIVACY DISCLAIMER</h4>
    <p style="margin-bottom: 0; color: #424242;">
        <strong>This demo is for research and educational purposes only.</strong><br>
        • DO NOT USE for biometric identification or voice cloning<br>
        • DO NOT USE for impersonation or malicious purposes<br>
        • Audio data is processed locally and not stored<br>
        • Respect privacy rights and data protection regulations
    </p>
</div>
""", unsafe_allow_html=True)

# Title
st.title("🎤 Low-Resource Speech Recognition Demo")
st.markdown("A modern implementation of speech recognition using transfer learning and advanced ASR techniques.")

# Sidebar
st.sidebar.header("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Wav2Vec2", "Conformer"],
    help="Choose the ASR model architecture"
)

# Device selection
device_options = ["auto", "cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_options.append("mps")

device = st.sidebar.selectbox(
    "Select Device",
    device_options,
    help="Choose the computation device"
)

# Model initialization
@st.cache_resource
def load_model(model_type: str, device: str):
    """Load the selected model."""
    try:
        if model_type == "Wav2Vec2":
            model = Wav2Vec2ASR(
                model_name="facebook/wav2vec2-base-960h",
                device=device
            )
        elif model_type == "Conformer":
            model = ConformerASR(
                input_dim=80,
                encoder_dim=512,
                num_encoder_layers=6,
                vocab_size=1000,
                device=device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
with st.spinner("Loading model..."):
    model = load_model(model_type, device)

if model is None:
    st.error("Failed to load model. Please check the configuration.")
    st.stop()

# Display model info
with st.expander("Model Information"):
    model_info = model.get_model_info()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Type:**", model_info.get("model_type", model_type))
        st.write("**Device:**", model_info.get("device", device))
        st.write("**Vocabulary Size:**", model_info.get("vocab_size", "N/A"))
    
    with col2:
        params = model_info.get("parameters", {})
        st.write("**Total Parameters:**", f"{params.get('total', 0):,}")
        st.write("**Trainable Parameters:**", f"{params.get('trainable', 0):,}")
        st.write("**Frozen Parameters:**", f"{params.get('frozen', 0):,}")

# Main content
st.header("Audio Input")

# Audio input options
input_method = st.radio(
    "Choose input method:",
    ["Upload Audio File", "Record Audio"],
    horizontal=True
)

audio_data = None
sample_rate = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
    )
    
    if uploaded_file is not None:
        try:
            # Load audio file
            audio_data, sample_rate = torchaudio.load(uploaded_file)
            
            # Convert to mono if stereo
            if audio_data.shape[0] > 1:
                audio_data = audio_data.mean(dim=0, keepdim=True)
            
            audio_data = audio_data.squeeze(0)  # Remove channel dimension
            
            st.success(f"Audio loaded successfully! Duration: {len(audio_data) / sample_rate:.2f} seconds")
            
        except Exception as e:
            st.error(f"Error loading audio file: {str(e)}")

elif input_method == "Record Audio":
    st.info("Audio recording functionality would be implemented here using Streamlit's audio recording capabilities.")
    st.markdown("For now, please use the file upload option.")

# Process audio if available
if audio_data is not None and sample_rate is not None:
    st.header("Audio Analysis")
    
    # Display audio waveform
    fig, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    ax.plot(time_axis, audio_data.numpy())
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Display spectrogram
    st.subheader("Spectrogram")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to numpy for librosa
    audio_np = audio_data.numpy()
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
    
    # Display spectrogram
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sample_rate, ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)
    
    # Speech recognition
    st.header("Speech Recognition")
    
    if st.button("Transcribe Audio", type="primary"):
        with st.spinner("Transcribing audio..."):
            try:
                # Transcribe audio
                if hasattr(model, 'transcribe'):
                    result = model.transcribe(audio_data, sample_rate, return_confidence=True)
                    if isinstance(result, tuple):
                        transcription, confidence = result
                    else:
                        transcription = result
                        confidence = 1.0
                else:
                    # Fallback for models without transcribe method
                    st.warning("Model does not support direct transcription. Using fallback method.")
                    transcription = "Transcription not available for this model type."
                    confidence = 0.0
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Transcription")
                    st.text_area(
                        "Recognized Text",
                        value=transcription,
                        height=100,
                        disabled=True
                    )
                
                with col2:
                    st.subheader("Confidence")
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.3f}",
                        help="Confidence score for the transcription (0-1)"
                    )
                    
                    # Confidence bar
                    st.progress(confidence)
                
                # Additional metrics
                st.subheader("Audio Information")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Duration", f"{len(audio_data) / sample_rate:.2f}s")
                
                with col2:
                    st.metric("Sample Rate", f"{sample_rate:,} Hz")
                
                with col3:
                    st.metric("Samples", f"{len(audio_data):,}")
                
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")
                logger.error(f"Transcription error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Low-Resource Speech Recognition Demo | Research & Educational Use Only</p>
    <p>Built with Streamlit, PyTorch, and Transformers</p>
</div>
""", unsafe_allow_html=True)
