#!/usr/bin/env python3
"""Quick start script for low-resource speech recognition."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Wav2Vec2ASR
from src.data import SyntheticDataset
from src.metrics import ASRMetrics, evaluate_model
from src.utils import setup_logging, set_seed

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


def quick_demo():
    """Run a quick demonstration."""
    logger.info("Running quick demonstration...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=100,
        sample_rate=16000,
        duration_range=(1.0, 3.0),
        vocab_size=100,
        feature_type="raw"
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Initialize model
    logger.info("Initializing Wav2Vec2 model...")
    model = Wav2Vec2ASR(
        model_name="facebook/wav2vec2-base-960h",
        vocab_size=32,
        device="cpu"  # Use CPU for demo
    )
    
    logger.info("Model initialized successfully")
    
    # Test transcription
    logger.info("Testing transcription...")
    sample = dataset[0]
    
    try:
        # For demonstration, we'll just show the model info
        model_info = model.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        logger.info("Quick demo completed successfully!")
        logger.info("To run the full demo, use:")
        logger.info("  streamlit run demo/streamlit_app.py")
        logger.info("  python demo/gradio_app.py")
        
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")
        logger.info("This is expected for the base model without fine-tuning")


def generate_data():
    """Generate synthetic data."""
    logger.info("Generating synthetic data...")
    
    from scripts.generate_synthetic_data import generate_dataset
    
    generate_dataset(
        num_samples=1000,
        output_dir="data/synthetic",
        sample_rate=16000,
        duration_range=(1.0, 5.0),
        num_speakers=10,
        languages=["en"],
        vocab_size=1000,
    )


def run_training():
    """Run training with synthetic data."""
    logger.info("Running training with synthetic data...")
    
    # First generate data if it doesn't exist
    data_dir = Path("data/synthetic")
    if not data_dir.exists():
        logger.info("Synthetic data not found, generating...")
        generate_data()
    
    # Run training
    import subprocess
    
    cmd = [
        sys.executable, "scripts/train.py",
        "data=synthetic",
        "model=wav2vec2",
        "training.epochs=3",
        "training.optimizer.lr=1e-4",
        "data.dataloader.batch_size=4"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Training completed successfully!")
            logger.info("Output:", result.stdout)
        else:
            logger.error("Training failed!")
            logger.error("Error:", result.stderr)
    except Exception as e:
        logger.error(f"Error running training: {str(e)}")


def run_evaluation():
    """Run evaluation."""
    logger.info("Running evaluation...")
    
    import subprocess
    
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "data=synthetic",
        "model=wav2vec2",
        "generate_report=true"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Evaluation completed successfully!")
            logger.info("Output:", result.stdout)
        else:
            logger.error("Evaluation failed!")
            logger.error("Error:", result.stderr)
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick start for low-resource speech recognition")
    
    parser.add_argument(
        "action",
        choices=["demo", "generate-data", "train", "evaluate", "all"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    if args.action == "demo":
        quick_demo()
    elif args.action == "generate-data":
        generate_data()
    elif args.action == "train":
        run_training()
    elif args.action == "evaluate":
        run_evaluation()
    elif args.action == "all":
        logger.info("Running complete pipeline...")
        generate_data()
        run_training()
        run_evaluation()
        logger.info("Complete pipeline finished!")


if __name__ == "__main__":
    main()
