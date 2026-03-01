"""Data loading and preprocessing for speech recognition."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from ..features import AudioPreprocessor, extract_features
from ..utils import sanitize_filename

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Dataset for audio speech recognition."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str = "meta.csv",
        audio_dir: str = "wav",
        feature_type: str = "log_mel",
        sample_rate: int = 16000,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
        normalize: bool = True,
        preemphasis: Optional[float] = None,
        trim_silence: bool = False,
        processor: Optional[Wav2Vec2Processor] = None,
        privacy_mode: bool = True,
        **feature_kwargs
    ):
        """Initialize audio dataset.
        
        Args:
            data_dir: Directory containing dataset
            meta_file: Name of metadata CSV file
            audio_dir: Name of audio files directory
            feature_type: Type of features to extract
            sample_rate: Target sample rate
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            normalize: Whether to normalize audio
            preemphasis: Preemphasis coefficient
            trim_silence: Whether to trim silence
            processor: Wav2Vec2 processor for tokenization
            privacy_mode: Whether to enable privacy protection
            **feature_kwargs: Additional arguments for feature extraction
        """
        self.data_dir = Path(data_dir)
        self.meta_file = meta_file
        self.audio_dir = audio_dir
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.privacy_mode = privacy_mode
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            normalize=normalize,
            preemphasis=preemphasis,
            trim_silence=trim_silence,
        )
        
        # Initialize feature extractor
        self.feature_kwargs = feature_kwargs
        
        # Initialize processor
        self.processor = processor
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter by duration
        self.metadata = self._filter_by_duration()
        
        logger.info(f"Loaded dataset with {len(self.metadata)} samples")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        meta_path = self.data_dir / self.meta_file
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        metadata = pd.read_csv(meta_path)
        
        # Validate required columns
        required_columns = ["id", "path", "text"]
        missing_columns = [col for col in required_columns if col not in metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add full paths
        metadata["full_path"] = metadata["path"].apply(
            lambda x: self.data_dir / self.audio_dir / x
        )
        
        # Check if audio files exist
        missing_files = []
        for idx, row in metadata.iterrows():
            if not row["full_path"].exists():
                missing_files.append(row["path"])
        
        if missing_files:
            logger.warning(f"Missing audio files: {missing_files[:5]}...")
            metadata = metadata[metadata["full_path"].apply(lambda x: x.exists())]
        
        return metadata
    
    def _filter_by_duration(self) -> pd.DataFrame:
        """Filter samples by duration."""
        if "duration" not in self.metadata.columns:
            logger.warning("Duration column not found, skipping duration filtering")
            return self.metadata
        
        initial_count = len(self.metadata)
        
        # Filter by duration
        mask = (
            (self.metadata["duration"] >= self.min_duration) &
            (self.metadata["duration"] <= self.max_duration)
        )
        
        self.metadata = self.metadata[mask]
        
        filtered_count = initial_count - len(self.metadata)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} samples by duration")
        
        return self.metadata
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = row["full_path"]
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        # Preprocess audio
        waveform = self.preprocessor.preprocess(waveform, sample_rate)
        
        # Extract features
        if self.feature_type == "raw":
            features = waveform
        else:
            features = extract_features(
                waveform,
                feature_type=self.feature_type,
                sample_rate=self.sample_rate,
                **self.feature_kwargs
            )
        
        # Prepare text
        text = str(row["text"]).strip()
        
        # Process with Wav2Vec2 processor if available
        if self.processor is not None:
            # Tokenize text
            labels = self.processor.tokenizer(text).input_ids
            
            # Process audio
            inputs = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            return {
                "input_values": inputs.input_values.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0),
                "labels": torch.tensor(labels, dtype=torch.long),
                "text": text,
                "id": row["id"],
                "duration": row.get("duration", 0.0),
            }
        
        # Return raw features
        return {
            "features": features.squeeze(0),
            "text": text,
            "id": row["id"],
            "duration": row.get("duration", 0.0),
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get sample information."""
        row = self.metadata.iloc[idx]
        
        info = {
            "id": row["id"],
            "text": row["text"],
            "duration": row.get("duration", 0.0),
            "path": row["path"],
        }
        
        # Add privacy-protected filename if enabled
        if self.privacy_mode:
            info["sanitized_path"] = sanitize_filename(row["path"])
        
        return info


class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration purposes."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        sample_rate: int = 16000,
        duration_range: Tuple[float, float] = (1.0, 5.0),
        vocab_size: int = 100,
        feature_type: str = "log_mel",
        **feature_kwargs
    ):
        """Initialize synthetic dataset.
        
        Args:
            num_samples: Number of synthetic samples
            sample_rate: Sample rate
            duration_range: Range of durations (min, max)
            vocab_size: Vocabulary size for text generation
            feature_type: Type of features to extract
            **feature_kwargs: Additional arguments for feature extraction
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration_range = duration_range
        self.vocab_size = vocab_size
        self.feature_type = feature_type
        self.feature_kwargs = feature_kwargs
        
        # Generate synthetic data
        self.samples = self._generate_samples()
        
        logger.info(f"Generated {len(self.samples)} synthetic samples")
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic samples."""
        import random
        import string
        
        samples = []
        
        for i in range(self.num_samples):
            # Generate random duration
            duration = random.uniform(*self.duration_range)
            num_samples = int(duration * self.sample_rate)
            
            # Generate synthetic audio (sine wave + noise)
            t = torch.linspace(0, duration, num_samples)
            freq = random.uniform(200, 800)  # Random frequency
            audio = torch.sin(2 * torch.pi * freq * t)
            audio += 0.1 * torch.randn_like(audio)  # Add noise
            
            # Generate synthetic text
            text_length = random.randint(5, 20)
            text = " ".join([
                "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
                for _ in range(text_length)
            ])
            
            samples.append({
                "audio": audio,
                "text": text,
                "duration": duration,
                "id": f"synthetic_{i:06d}",
            })
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        sample = self.samples[idx]
        
        # Extract features
        if self.feature_type == "raw":
            features = sample["audio"]
        else:
            features = extract_features(
                sample["audio"].unsqueeze(0),
                feature_type=self.feature_type,
                sample_rate=self.sample_rate,
                **self.feature_kwargs
            ).squeeze(0)
        
        return {
            "features": features,
            "text": sample["text"],
            "id": sample["id"],
            "duration": sample["duration"],
        }


def create_data_splits(
    dataset: AudioDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: Optional[str] = None,
    random_state: int = 42
) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
    """Create train/validation/test splits.
    
    Args:
        dataset: Input dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify_by: Column to stratify by
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of train, validation, and test datasets
    """
    from sklearn.model_selection import train_test_split
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Get indices
    indices = list(range(len(dataset)))
    
    if stratify_by and stratify_by in dataset.metadata.columns:
        # Stratified split
        stratify = dataset.metadata[stratify_by].values
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=stratify,
            random_state=random_state
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            stratify=stratify[temp_idx] if stratify_by else None,
            random_state=random_state
        )
    else:
        # Random split
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_state
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            random_state=random_state
        )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    logger.info(f"Created splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset
