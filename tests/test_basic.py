"""Test suite for low-resource speech recognition."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Wav2Vec2ASR, ConformerASR
from src.data import SyntheticDataset
from src.features import MelSpectrogram, MFCC, SpecAugment, AudioPreprocessor
from src.metrics import ASRMetrics, ConfidenceCalibration, PerformanceMetrics
from src.utils import set_seed, get_device, sanitize_filename


class TestModels:
    """Test model implementations."""
    
    def test_wav2vec2_initialization(self):
        """Test Wav2Vec2 model initialization."""
        model = Wav2Vec2ASR(
            model_name="facebook/wav2vec2-base-960h",
            vocab_size=32,
            device="cpu"
        )
        
        assert model is not None
        assert model.vocab_size == 32
        assert str(model.device_manager.device) == "cpu"
    
    def test_conformer_initialization(self):
        """Test Conformer model initialization."""
        model = ConformerASR(
            input_dim=80,
            encoder_dim=512,
            num_encoder_layers=2,  # Small for testing
            vocab_size=100,
            device="cpu"
        )
        
        assert model is not None
        assert model.input_dim == 80
        assert model.encoder_dim == 512
        assert model.vocab_size == 100
    
    def test_conformer_forward(self):
        """Test Conformer forward pass."""
        model = ConformerASR(
            input_dim=80,
            encoder_dim=512,
            num_encoder_layers=2,
            vocab_size=100,
            device="cpu"
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 100
        input_dim = 80
        
        features = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        outputs = model(features)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 100)
        assert outputs["loss"] is None  # No labels provided


class TestData:
    """Test data loading and preprocessing."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticDataset(
            num_samples=10,
            sample_rate=16000,
            duration_range=(1.0, 2.0),
            vocab_size=50,
            feature_type="raw"
        )
        
        assert len(dataset) == 10
        
        # Test getting a sample
        sample = dataset[0]
        
        assert "features" in sample
        assert "text" in sample
        assert "id" in sample
        assert "duration" in sample
        
        assert isinstance(sample["features"], torch.Tensor)
        assert isinstance(sample["text"], str)
        assert isinstance(sample["id"], str)
        assert isinstance(sample["duration"], float)
    
    def test_synthetic_dataset_with_features(self):
        """Test synthetic dataset with feature extraction."""
        dataset = SyntheticDataset(
            num_samples=5,
            sample_rate=16000,
            duration_range=(1.0, 2.0),
            vocab_size=50,
            feature_type="log_mel",
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        assert len(dataset) == 5
        
        # Test getting a sample
        sample = dataset[0]
        
        assert "features" in sample
        assert sample["features"].shape[0] == 80  # n_mels


class TestFeatures:
    """Test feature extraction."""
    
    def test_mel_spectrogram(self):
        """Test mel spectrogram extraction."""
        extractor = MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        
        # Create dummy audio
        audio = torch.randn(1, 16000)  # 1 second of audio
        
        # Extract features
        features = extractor(audio)
        
        assert features.shape[0] == 1  # batch size
        assert features.shape[1] == 80  # n_mels
        assert features.shape[2] > 0  # time dimension
    
    def test_mfcc(self):
        """Test MFCC extraction."""
        extractor = MFCC(
            sample_rate=16000,
            n_mfcc=13,
            n_fft=1024,
            hop_length=256
        )
        
        # Create dummy audio
        audio = torch.randn(1, 16000)  # 1 second of audio
        
        # Extract features
        features = extractor(audio)
        
        assert features.shape[0] == 1  # batch size
        assert features.shape[1] == 13  # n_mfcc
        assert features.shape[2] > 0  # time dimension
    
    def test_spec_augment(self):
        """Test SpecAugment."""
        augment = SpecAugment(
            time_mask_param=10,
            freq_mask_param=5,
            num_time_mask=1,
            num_freq_mask=1
        )
        
        # Create dummy spectrogram
        spec = torch.randn(1, 80, 100)  # batch, freq, time
        
        # Apply augmentation
        augmented = augment(spec)
        
        assert augmented.shape == spec.shape
    
    def test_audio_preprocessor(self):
        """Test audio preprocessing."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            normalize=True,
            preemphasis=0.97,
            trim_silence=False
        )
        
        # Create dummy audio
        audio = torch.randn(1, 16000)
        
        # Preprocess
        processed = preprocessor.preprocess(audio, 16000)
        
        assert processed.shape == audio.shape
        assert torch.max(torch.abs(processed)) <= 1.0  # Normalized


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_asr_metrics(self):
        """Test ASR metrics calculation."""
        metrics = ASRMetrics()
        
        predictions = ["hello world", "good morning", "how are you"]
        references = ["hello world", "good evening", "how are you"]
        
        # Test WER
        wer = metrics.word_error_rate(predictions, references)
        assert isinstance(wer, float)
        assert 0 <= wer <= 1
        
        # Test CER
        cer = metrics.character_error_rate(predictions, references)
        assert isinstance(cer, float)
        assert 0 <= cer <= 1
        
        # Test token accuracy
        token_acc = metrics.token_accuracy(predictions, references)
        assert isinstance(token_acc, float)
        assert 0 <= token_acc <= 1
        
        # Test all metrics
        all_metrics = metrics.compute_all_metrics(predictions, references)
        assert isinstance(all_metrics, dict)
        assert "wer" in all_metrics
        assert "cer" in all_metrics
        assert "token_accuracy" in all_metrics
    
    def test_confidence_calibration(self):
        """Test confidence calibration metrics."""
        calibration = ConfidenceCalibration(num_bins=5)
        
        confidences = [0.8, 0.9, 0.7, 0.6, 0.5]
        accuracies = [True, True, False, False, False]
        
        # Test ECE
        ece = calibration.expected_calibration_error(confidences, accuracies)
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
        
        # Test MCE
        mce = calibration.maximum_calibration_error(confidences, accuracies)
        assert isinstance(mce, float)
        assert 0 <= mce <= 1
    
    def test_performance_metrics(self):
        """Test performance metrics."""
        perf = PerformanceMetrics()
        
        # Add some samples
        perf.add_sample(1.0, 0.5)  # 1 second audio, 0.5 second inference
        perf.add_sample(2.0, 1.0)  # 2 second audio, 1.0 second inference
        
        # Test RTF
        rtf = perf.real_time_factor()
        assert isinstance(rtf, float)
        assert rtf > 0
        
        # Test throughput
        throughput = perf.throughput()
        assert isinstance(throughput, float)
        assert throughput > 0
        
        # Test all metrics
        all_metrics = perf.get_all_metrics()
        assert isinstance(all_metrics, dict)
        assert "real_time_factor" in all_metrics
        assert "throughput" in all_metrics


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        # Should be the same
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert str(device) == "cpu"
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test with date
        filename = "recording_2023-12-01.wav"
        sanitized = sanitize_filename(filename)
        assert sanitized.endswith(".wav")
        assert "DATE" in sanitized or len(sanitized) < len(filename)
        
        # Test with email
        filename = "user@example.com_audio.wav"
        sanitized = sanitize_filename(filename)
        assert sanitized.endswith(".wav")
        assert "EMAIL" in sanitized or len(sanitized) < len(filename)
        
        # Test with phone
        filename = "call_555-123-4567.wav"
        sanitized = sanitize_filename(filename)
        assert sanitized.endswith(".wav")
        assert "PHONE" in sanitized or len(sanitized) < len(filename)


if __name__ == "__main__":
    pytest.main([__file__])
