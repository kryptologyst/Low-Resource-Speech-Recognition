"""Audio feature extraction utilities."""

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from torch import nn

logger = logging.getLogger(__name__)


class MelSpectrogram(nn.Module):
    """Mel spectrogram feature extractor."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        normalized: bool = False,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        """Initialize mel spectrogram extractor.
        
        Args:
            sample_rate: Sample rate of input audio
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window size
            n_mels: Number of mel filterbanks
            fmin: Minimum frequency
            fmax: Maximum frequency
            window: Window function type
            center: Whether to pad signals
            pad_mode: Padding mode
            power: Exponent for magnitude spectrogram
            normalized: Whether to normalize
            norm: Normalization method
            mel_scale: Mel scale type
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalized = normalized
        
        # Create mel filterbank
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=fmin,
            f_max=self.fmax,
            n_stft=n_fft // 2 + 1,
            norm=norm,
            mel_scale=mel_scale,
        )
        
        # Create spectrogram transform
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=getattr(torch, f"{window}_window"),
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features.
        
        Args:
            waveform: Input waveform tensor [batch, samples]
            
        Returns:
            torch.Tensor: Mel spectrogram [batch, n_mels, time]
        """
        # Compute spectrogram
        spec = self.spectrogram(waveform)
        
        # Convert to mel scale
        mel_spec = self.mel_scale(spec)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-8)
        
        return mel_spec


class MFCC(nn.Module):
    """MFCC feature extractor."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        power: float = 2.0,
        normalized: bool = False,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        """Initialize MFCC extractor.
        
        Args:
            sample_rate: Sample rate of input audio
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            win_length: Window size
            n_mels: Number of mel filterbanks
            fmin: Minimum frequency
            fmax: Maximum frequency
            window: Window function type
            center: Whether to pad signals
            pad_mode: Padding mode
            power: Exponent for magnitude spectrogram
            normalized: Whether to normalize
            norm: Normalization method
            mel_scale: Mel scale type
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalized = normalized
        
        # Create MFCC transform
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "window_fn": getattr(torch, f"{window}_window"),
                "power": power,
                "normalized": normalized,
                "center": center,
                "pad_mode": pad_mode,
                "n_mels": n_mels,
                "f_min": fmin,
                "f_max": self.fmax,
                "norm": norm,
                "mel_scale": mel_scale,
            }
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features.
        
        Args:
            waveform: Input waveform tensor [batch, samples]
            
        Returns:
            torch.Tensor: MFCC features [batch, n_mfcc, time]
        """
        return self.mfcc(waveform)


class SpecAugment(nn.Module):
    """SpecAugment data augmentation for speech recognition."""
    
    def __init__(
        self,
        time_mask_param: int = 27,
        freq_mask_param: int = 12,
        num_time_mask: int = 2,
        num_freq_mask: int = 2,
        p: float = 1.0,
    ):
        """Initialize SpecAugment.
        
        Args:
            time_mask_param: Maximum time mask length
            freq_mask_param: Maximum frequency mask length
            num_time_mask: Number of time masks
            num_freq_mask: Number of frequency masks
            p: Probability of applying augmentation
        """
        super().__init__()
        
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_mask = num_time_mask
        self.num_freq_mask = num_freq_mask
        self.p = p
        
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param,
            p=p,
        )
        
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param,
            p=p,
        )
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.
        
        Args:
            spec: Input spectrogram [batch, freq, time]
            
        Returns:
            torch.Tensor: Augmented spectrogram
        """
        # Apply frequency masking
        for _ in range(self.num_freq_mask):
            spec = self.freq_mask(spec)
        
        # Apply time masking
        for _ in range(self.num_time_mask):
            spec = self.time_mask(spec)
        
        return spec


class AudioPreprocessor:
    """Audio preprocessing utilities."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        preemphasis: Optional[float] = None,
        trim_silence: bool = False,
        trim_threshold: float = 0.01,
    ):
        """Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            normalize: Whether to normalize audio
            preemphasis: Preemphasis coefficient
            trim_silence: Whether to trim silence
            trim_threshold: Silence threshold for trimming
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.preemphasis = preemphasis
        self.trim_silence = trim_silence
        self.trim_threshold = trim_threshold
    
    def preprocess(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Preprocess audio waveform.
        
        Args:
            waveform: Input waveform
            orig_sr: Original sample rate
            
        Returns:
            torch.Tensor: Preprocessed waveform
        """
        # Resample if necessary
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate
            )
        
        # Trim silence
        if self.trim_silence:
            waveform = self._trim_silence(waveform)
        
        # Apply preemphasis
        if self.preemphasis is not None:
            waveform = self._apply_preemphasis(waveform)
        
        # Normalize
        if self.normalize:
            waveform = self._normalize(waveform)
        
        return waveform
    
    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim silence from waveform."""
        # Convert to numpy for librosa
        audio_np = waveform.squeeze().numpy()
        
        # Trim silence
        trimmed, _ = librosa.effects.trim(
            audio_np, top_db=20, frame_length=2048, hop_length=512
        )
        
        return torch.from_numpy(trimmed).unsqueeze(0)
    
    def _apply_preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply preemphasis filter."""
        return torchaudio.functional.preemphasis(waveform, self.preemphasis)
    
    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize waveform."""
        return waveform / (torch.max(torch.abs(waveform)) + 1e-8)


def extract_features(
    waveform: torch.Tensor,
    feature_type: str = "log_mel",
    sample_rate: int = 16000,
    **kwargs
) -> torch.Tensor:
    """Extract audio features from waveform.
    
    Args:
        waveform: Input waveform tensor
        feature_type: Type of features to extract
        sample_rate: Sample rate of audio
        **kwargs: Additional arguments for feature extraction
        
    Returns:
        torch.Tensor: Extracted features
    """
    if feature_type == "log_mel":
        extractor = MelSpectrogram(sample_rate=sample_rate, **kwargs)
    elif feature_type == "mfcc":
        extractor = MFCC(sample_rate=sample_rate, **kwargs)
    elif feature_type == "raw":
        return waveform
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return extractor(waveform)
