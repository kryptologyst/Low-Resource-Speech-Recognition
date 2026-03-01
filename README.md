# Low-Resource Speech Recognition

Research-focused implementation of low-resource speech recognition using transfer learning and advanced ASR techniques.

## PRIVACY DISCLAIMER

**IMPORTANT: This project is for research and educational purposes only.**

- This software is designed for academic research and educational demonstrations
- **DO NOT USE** for biometric identification, voice cloning, or any production biometric applications
- **DO NOT USE** for impersonation, deepfake generation, or any malicious purposes
- Users are responsible for ensuring compliance with local laws and regulations
- The authors disclaim any responsibility for misuse of this software
- Always obtain proper consent before processing any audio data
- Respect privacy rights and data protection regulations

## Overview

This project implements state-of-the-art low-resource speech recognition techniques including:

- Transfer learning with pre-trained Wav2Vec 2.0 models
- Conformer-based architectures for improved accuracy
- CTC and attention-based decoding strategies
- Data augmentation techniques for low-resource scenarios
- Comprehensive evaluation metrics (WER, CER, confidence calibration)
- Interactive demo interface for real-time testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Low-Resource-Speech-Recognition.git
cd Low-Resource-Speech-Recognition

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import Wav2Vec2ASR
from src.data import AudioDataset

# Load pre-trained model
model = Wav2Vec2ASR.from_pretrained("facebook/wav2vec2-large-960h")

# Fine-tune on your low-resource language data
model.fine_tune(train_dataset, validation_dataset)

# Transcribe audio
transcription = model.transcribe("path/to/audio.wav")
print(f"Transcription: {transcription}")
```

### Demo Interface

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py

# Or launch Gradio demo
python demo/gradio_app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature extraction
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── decoding/          # Decoding strategies
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation utilities
│   └── utils/             # General utilities
├── data/                  # Data directory
│   ├── raw/              # Raw audio files
│   └── processed/        # Processed datasets
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit tests
├── assets/               # Generated artifacts
├── demo/                 # Demo applications
└── docs/                 # Documentation
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml` - Main configuration
- `configs/model/` - Model-specific configurations
- `configs/data/` - Dataset configurations
- `configs/training/` - Training configurations

## Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py model=conformer data=low_resource_lang training.epochs=50

# Resume training from checkpoint
python scripts/train.py training.resume_from_checkpoint=checkpoints/latest.ckpt
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py model=checkpoints/best_model.ckpt data=test_set

# Generate evaluation report
python scripts/evaluate.py --generate_report
```

## Dataset Schema

The project expects audio datasets in the following format:

```
data/
├── meta.csv              # Dataset metadata
├── wav/                  # Audio files directory
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── annotations.json      # Optional: detailed annotations
```

### Meta.csv Format

| column | description | example |
|--------|-------------|---------|
| id | Unique identifier | "sample_001" |
| path | Relative path to audio file | "wav/sample_001.wav" |
| sr | Sample rate | 16000 |
| duration | Duration in seconds | 3.45 |
| text | Transcription text | "hello world" |
| language | Language code | "en" |
| speaker_id | Speaker identifier | "spk_001" |
| split | Data split | "train" |

## Models

### Supported Architectures

1. **Wav2Vec 2.0** - Pre-trained transformer-based encoder
2. **Conformer** - Convolution-augmented transformer
3. **CTC/Attention Hybrid** - Combined CTC and attention decoding

### Pre-trained Models

- `facebook/wav2vec2-large-960h` - English (960h training)
- `facebook/wav2vec2-large-960h-lv60-self` - Multilingual
- `facebook/wav2vec2-xlsr-53` - Cross-lingual representation

## Evaluation Metrics

### ASR Metrics
- **WER (Word Error Rate)** - Primary metric for ASR
- **CER (Character Error Rate)** - Character-level accuracy
- **Token Accuracy** - Token-level precision
- **Confidence Calibration** - Reliability of confidence scores

### Performance Metrics
- **Latency** - Inference time per utterance
- **RTF (Real-Time Factor)** - Processing speed relative to audio duration
- **Memory Usage** - Peak memory consumption

## Data Augmentation

The project includes comprehensive data augmentation techniques:

- **SpecAugment** - Time and frequency masking
- **Speed Perturbation** - Time stretching
- **Pitch Shifting** - Frequency modification
- **Noise Injection** - Background noise addition
- **Reverb Simulation** - Room impulse response convolution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{low_resource_speech_recognition,
  title={Low-Resource Speech Recognition},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Low-Resource-Speech-Recognition}
}
```

## Acknowledgments

- Hugging Face Transformers team for the excellent Wav2Vec 2.0 implementation
- Mozilla Common Voice for providing open speech datasets
- The speech recognition research community for continuous advances

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`
# Low-Resource-Speech-Recognition
