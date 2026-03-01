#!/usr/bin/env python3
"""Generate synthetic dataset for demonstration purposes."""

import argparse
import json
import logging
import random
import string
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_audio(
    duration: float,
    sample_rate: int = 16000,
    complexity: str = "simple"
) -> torch.Tensor:
    """Generate synthetic audio.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        complexity: Complexity level ("simple", "medium", "complex")
        
    Returns:
        torch.Tensor: Generated audio
    """
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)
    
    if complexity == "simple":
        # Simple sine wave
        freq = random.uniform(200, 800)
        audio = torch.sin(2 * torch.pi * freq * t)
        
    elif complexity == "medium":
        # Multiple sine waves with harmonics
        audio = torch.zeros_like(t)
        base_freq = random.uniform(200, 400)
        
        for harmonic in range(1, 4):
            freq = base_freq * harmonic
            amplitude = 1.0 / harmonic
            audio += amplitude * torch.sin(2 * torch.pi * freq * t)
        
        # Normalize
        audio = audio / torch.max(torch.abs(audio))
        
    else:  # complex
        # Complex waveform with noise and modulation
        audio = torch.zeros_like(t)
        base_freq = random.uniform(200, 400)
        
        # Add multiple components
        for i in range(5):
            freq = base_freq * (1 + i * 0.5)
            amplitude = random.uniform(0.1, 0.5)
            phase = random.uniform(0, 2 * torch.pi)
            audio += amplitude * torch.sin(2 * torch.pi * freq * t + phase)
        
        # Add frequency modulation
        mod_freq = random.uniform(0.5, 2.0)
        mod_depth = random.uniform(0.1, 0.3)
        audio *= (1 + mod_depth * torch.sin(2 * torch.pi * mod_freq * t))
        
        # Add noise
        noise_level = random.uniform(0.05, 0.15)
        audio += noise_level * torch.randn_like(audio)
        
        # Normalize
        audio = audio / torch.max(torch.abs(audio))
    
    return audio


def generate_synthetic_text(
    length_range: tuple = (3, 15),
    vocab_size: int = 1000,
    language: str = "en"
) -> str:
    """Generate synthetic text.
    
    Args:
        length_range: Range of word lengths
        vocab_size: Vocabulary size
        language: Language code
        
    Returns:
        str: Generated text
    """
    if language == "en":
        # Generate English-like words
        words = []
        num_words = random.randint(*length_range)
        
        for _ in range(num_words):
            word_length = random.randint(3, 8)
            word = "".join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
        
        return " ".join(words)
    
    else:
        # Generate random tokens for other languages
        tokens = []
        num_tokens = random.randint(*length_range)
        
        for _ in range(num_tokens):
            token = f"token_{random.randint(1, vocab_size)}"
            tokens.append(token)
        
        return " ".join(tokens)


def create_speaker_characteristics(speaker_id: str) -> Dict[str, Any]:
    """Create speaker characteristics.
    
    Args:
        speaker_id: Speaker identifier
        
    Returns:
        Dict containing speaker characteristics
    """
    # Generate consistent characteristics based on speaker ID
    random.seed(hash(speaker_id) % 2**32)
    
    return {
        "speaker_id": speaker_id,
        "gender": random.choice(["male", "female", "other"]),
        "age_group": random.choice(["child", "adult", "elderly"]),
        "accent": random.choice(["native", "regional", "foreign"]),
        "speaking_rate": random.uniform(0.8, 1.2),
        "pitch_range": random.uniform(0.7, 1.3),
    }


def generate_dataset(
    num_samples: int = 1000,
    output_dir: str = "data/synthetic",
    sample_rate: int = 16000,
    duration_range: tuple = (1.0, 5.0),
    complexity_distribution: Dict[str, float] = None,
    num_speakers: int = 10,
    languages: List[str] = None,
    vocab_size: int = 1000,
) -> None:
    """Generate synthetic dataset.
    
    Args:
        num_samples: Number of samples to generate
        output_dir: Output directory
        sample_rate: Sample rate
        duration_range: Range of durations
        complexity_distribution: Distribution of complexity levels
        num_speakers: Number of speakers
        languages: List of languages
        vocab_size: Vocabulary size
    """
    if complexity_distribution is None:
        complexity_distribution = {"simple": 0.4, "medium": 0.4, "complex": 0.2}
    
    if languages is None:
        languages = ["en"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_dir = output_path / "wav"
    audio_dir.mkdir(exist_ok=True)
    
    # Generate speaker IDs
    speakers = [f"speaker_{i:03d}" for i in range(num_speakers)]
    
    # Generate samples
    samples = []
    speaker_characteristics = {}
    
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    for i in tqdm(range(num_samples)):
        # Generate sample ID
        sample_id = f"synthetic_{i:06d}"
        
        # Select speaker
        speaker_id = random.choice(speakers)
        
        # Get or create speaker characteristics
        if speaker_id not in speaker_characteristics:
            speaker_characteristics[speaker_id] = create_speaker_characteristics(speaker_id)
        
        # Generate duration
        duration = random.uniform(*duration_range)
        
        # Select complexity
        complexity = np.random.choice(
            list(complexity_distribution.keys()),
            p=list(complexity_distribution.values())
        )
        
        # Select language
        language = random.choice(languages)
        
        # Generate audio
        audio = generate_synthetic_audio(duration, sample_rate, complexity)
        
        # Generate text
        text = generate_synthetic_text(
            length_range=(3, 15),
            vocab_size=vocab_size,
            language=language
        )
        
        # Save audio file
        audio_filename = f"{sample_id}.wav"
        audio_path = audio_dir / audio_filename
        
        torchaudio.save(
            str(audio_path),
            audio.unsqueeze(0),
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )
        
        # Create sample metadata
        sample = {
            "id": sample_id,
            "path": f"wav/{audio_filename}",
            "text": text,
            "duration": duration,
            "sample_rate": sample_rate,
            "language": language,
            "speaker_id": speaker_id,
            "complexity": complexity,
            "split": "train" if i < num_samples * 0.8 else "val" if i < num_samples * 0.9 else "test"
        }
        
        samples.append(sample)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(samples)
    
    # Save metadata
    metadata_path = output_path / "meta.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    # Save speaker characteristics
    speaker_path = output_path / "speakers.json"
    with open(speaker_path, "w") as f:
        json.dump(speaker_characteristics, f, indent=2)
    
    # Create dataset info
    dataset_info = {
        "name": "synthetic_speech_dataset",
        "version": "1.0.0",
        "description": "Synthetic speech dataset for demonstration purposes",
        "num_samples": num_samples,
        "num_speakers": num_speakers,
        "languages": languages,
        "sample_rate": sample_rate,
        "duration_range": duration_range,
        "complexity_distribution": complexity_distribution,
        "vocab_size": vocab_size,
        "splits": {
            "train": len(metadata_df[metadata_df["split"] == "train"]),
            "val": len(metadata_df[metadata_df["split"] == "val"]),
            "test": len(metadata_df[metadata_df["split"] == "test"])
        }
    }
    
    info_path = output_path / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Print summary
    logger.info("Dataset generation completed!")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Speakers: {num_speakers}")
    logger.info(f"Languages: {languages}")
    logger.info(f"Train/Val/Test: {dataset_info['splits']['train']}/{dataset_info['splits']['val']}/{dataset_info['splits']['test']}")
    
    # Print file sizes
    total_size = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
    logger.info(f"Total audio size: {total_size / (1024**2):.2f} MB")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic speech dataset")
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate"
    )
    
    parser.add_argument(
        "--duration-range",
        type=float,
        nargs=2,
        default=[1.0, 5.0],
        help="Duration range in seconds"
    )
    
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=10,
        help="Number of speakers"
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en"],
        help="Languages to generate"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Vocabulary size"
    )
    
    parser.add_argument(
        "--complexity-distribution",
        type=str,
        default="0.4,0.4,0.2",
        help="Complexity distribution (simple,medium,complex)"
    )
    
    args = parser.parse_args()
    
    # Parse complexity distribution
    complexity_dist = dict(zip(
        ["simple", "medium", "complex"],
        [float(x) for x in args.complexity_distribution.split(",")]
    ))
    
    # Generate dataset
    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        duration_range=tuple(args.duration_range),
        complexity_distribution=complexity_dist,
        num_speakers=args.num_speakers,
        languages=args.languages,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
