"""Gradio demo for low-resource speech recognition."""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
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

# Global model variable
model = None

def load_model(model_type: str, device: str) -> Optional[object]:
    """Load the selected model."""
    global model
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
        logger.error(f"Error loading model: {str(e)}")
        return None

def transcribe_audio(audio_file, model_type: str, device: str) -> Tuple[str, str, str]:
    """Transcribe audio file."""
    global model
    
    if audio_file is None:
        return "No audio file provided", "", ""
    
    try:
        # Load model if not loaded or if type/device changed
        if model is None or model.__class__.__name__ != model_type:
            model = load_model(model_type, device)
        
        if model is None:
            return "Error loading model", "", ""
        
        # Load audio file
        audio_data, sample_rate = torchaudio.load(audio_file)
        
        # Convert to mono if stereo
        if audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0, keepdim=True)
        
        audio_data = audio_data.squeeze(0)  # Remove channel dimension
        
        # Transcribe audio
        if hasattr(model, 'transcribe'):
            result = model.transcribe(audio_data, sample_rate, return_confidence=True)
            if isinstance(result, tuple):
                transcription, confidence = result
            else:
                transcription = result
                confidence = 1.0
        else:
            transcription = "Transcription not available for this model type."
            confidence = 0.0
        
        # Create audio info
        duration = len(audio_data) / sample_rate
        audio_info = f"Duration: {duration:.2f}s | Sample Rate: {sample_rate:,} Hz | Samples: {len(audio_data):,}"
        
        # Create confidence info
        confidence_info = f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)"
        
        return transcription, audio_info, confidence_info
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return f"Error during transcription: {str(e)}", "", ""

def create_interface():
    """Create Gradio interface."""
    
    # Privacy disclaimer
    privacy_text = """
    **PRIVACY DISCLAIMER**
    
    This demo is for research and educational purposes only.
    • DO NOT USE for biometric identification or voice cloning
    • DO NOT USE for impersonation or malicious purposes
    • Audio data is processed locally and not stored
    • Respect privacy rights and data protection regulations
    """
    
    with gr.Blocks(
        title="Low-Resource Speech Recognition Demo",
        theme=gr.themes.Soft(),
        css="""
        .privacy-warning {
            background-color: #ffebee;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #f44336;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        gr.Markdown("# 🎤 Low-Resource Speech Recognition Demo")
        gr.Markdown("A modern implementation of speech recognition using transfer learning and advanced ASR techniques.")
        
        # Privacy warning
        gr.Markdown(f"""
        <div class="privacy-warning">
            <h4 style="color: #c62828; margin-top: 0;">{privacy_text}</h4>
        </div>
        """, elem_classes="privacy-warning")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model configuration
                gr.Markdown("### Model Configuration")
                
                model_type = gr.Dropdown(
                    choices=["Wav2Vec2", "Conformer"],
                    value="Wav2Vec2",
                    label="Model Type",
                    info="Choose the ASR model architecture"
                )
                
                device_options = ["auto", "cpu"]
                if torch.cuda.is_available():
                    device_options.append("cuda")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device_options.append("mps")
                
                device = gr.Dropdown(
                    choices=device_options,
                    value="auto",
                    label="Device",
                    info="Choose the computation device"
                )
                
                # Audio input
                gr.Markdown("### Audio Input")
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    info="Supported formats: WAV, MP3, FLAC, M4A, OGG"
                )
                
                transcribe_btn = gr.Button(
                    "Transcribe Audio",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Results
                gr.Markdown("### Transcription Results")
                
                transcription_output = gr.Textbox(
                    label="Recognized Text",
                    lines=5,
                    max_lines=10,
                    interactive=False,
                    placeholder="Transcription will appear here..."
                )
                
                with gr.Row():
                    audio_info_output = gr.Textbox(
                        label="Audio Information",
                        interactive=False,
                        placeholder="Audio information will appear here..."
                    )
                    
                    confidence_output = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                        placeholder="Confidence score will appear here..."
                    )
        
        # Model information
        with gr.Accordion("Model Information", open=False):
            model_info_text = gr.Markdown("Model information will be displayed here after loading.")
        
        # Examples
        with gr.Accordion("Examples", open=False):
            gr.Markdown("""
            ### Example Audio Files
            
            You can test the demo with sample audio files. Here are some suggestions:
            
            1. **Clear Speech**: Record yourself saying a simple sentence clearly
            2. **Noisy Environment**: Try with background noise
            3. **Different Languages**: Test with non-English audio
            4. **Long Audio**: Test with longer recordings
            
            **Note**: The model is primarily trained on English, so results may vary for other languages.
            """)
        
        # Event handlers
        def update_model_info(model_type, device):
            """Update model information display."""
            try:
                model = load_model(model_type, device)
                if model is not None:
                    model_info = model.get_model_info()
                    info_text = f"""
                    **Model Type:** {model_info.get('model_type', model_type)}
                    **Device:** {model_info.get('device', device)}
                    **Vocabulary Size:** {model_info.get('vocab_size', 'N/A')}
                    
                    **Parameters:**
                    - Total: {model_info.get('parameters', {}).get('total', 0):,}
                    - Trainable: {model_info.get('parameters', {}).get('trainable', 0):,}
                    - Frozen: {model_info.get('parameters', {}).get('frozen', 0):,}
                    """
                    return info_text
                else:
                    return "Error loading model information."
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Connect events
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, model_type, device],
            outputs=[transcription_output, audio_info_output, confidence_output]
        )
        
        model_type.change(
            fn=update_model_info,
            inputs=[model_type, device],
            outputs=[model_info_text]
        )
        
        device.change(
            fn=update_model_info,
            inputs=[model_type, device],
            outputs=[model_info_text]
        )
        
        # Initialize model info
        interface.load(
            fn=update_model_info,
            inputs=[model_type, device],
            outputs=[model_info_text]
        )
    
    return interface

def main():
    """Main function to launch the demo."""
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_tips=True,
        enable_queue=True,
        max_threads=4
    )

if __name__ == "__main__":
    main()
