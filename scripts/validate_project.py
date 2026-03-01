#!/usr/bin/env python3
"""Project summary and validation script."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Wav2Vec2ASR, ConformerASR
from src.data import SyntheticDataset
from src.metrics import ASRMetrics
from src.utils import setup_logging, set_seed, get_device

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


def validate_project():
    """Validate that the project is properly set up."""
    logger.info("Validating project setup...")
    
    # Check project structure
    required_dirs = [
        "src/models", "src/data", "src/features", "src/metrics", "src/utils",
        "configs", "scripts", "tests", "demo", "notebooks", "assets"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            logger.error(f"Missing required directory: {dir_path}")
            return False
        else:
            logger.info(f"✓ {dir_path}")
    
    # Check required files
    required_files = [
        "pyproject.toml", "README.md", ".gitignore", "configs/config.yaml",
        "scripts/train.py", "scripts/evaluate.py", "demo/streamlit_app.py",
        "demo/gradio_app.py", "tests/test_basic.py"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Missing required file: {file_path}")
            return False
        else:
            logger.info(f"✓ {file_path}")
    
    return True


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test core imports
        from src.models import Wav2Vec2ASR, ConformerASR
        from src.data import SyntheticDataset
        from src.features import MelSpectrogram, MFCC, SpecAugment
        from src.metrics import ASRMetrics, ConfidenceCalibration
        from src.utils import set_seed, get_device, sanitize_filename
        
        logger.info("✓ All core modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    logger.info("Testing basic functionality...")
    
    try:
        # Set seed
        set_seed(42)
        
        # Test device selection
        device = get_device("cpu")
        logger.info(f"✓ Device selection: {device}")
        
        # Test synthetic dataset
        dataset = SyntheticDataset(num_samples=5, feature_type="raw")
        logger.info(f"✓ Synthetic dataset: {len(dataset)} samples")
        
        # Test metrics
        metrics = ASRMetrics()
        predictions = ["hello world", "good morning"]
        references = ["hello world", "good evening"]
        wer = metrics.word_error_rate(predictions, references)
        logger.info(f"✓ WER calculation: {wer:.4f}")
        
        logger.info("✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Functionality test error: {e}")
        return False


def print_project_summary():
    """Print project summary."""
    print("\n" + "="*60)
    print("LOW-RESOURCE SPEECH RECOGNITION PROJECT")
    print("="*60)
    
    print("\n📁 PROJECT STRUCTURE:")
    print("├── src/                    # Source code")
    print("│   ├── models/            # Model architectures (Wav2Vec2, Conformer)")
    print("│   ├── data/              # Data loading and preprocessing")
    print("│   ├── features/          # Feature extraction (Mel, MFCC, SpecAugment)")
    print("│   ├── metrics/           # Evaluation metrics (WER, CER, confidence)")
    print("│   └── utils/             # Utility functions")
    print("├── configs/               # Hydra configuration files")
    print("├── scripts/               # Training and evaluation scripts")
    print("├── tests/                 # Unit tests")
    print("├── demo/                  # Interactive demos (Streamlit, Gradio)")
    print("├── notebooks/             # Jupyter notebooks")
    print("└── assets/                # Generated artifacts")
    
    print("\n🚀 QUICK START:")
    print("1. Install dependencies:")
    print("   pip install -e .")
    print("\n2. Run quick demo:")
    print("   python scripts/quick_start.py demo")
    print("\n3. Generate synthetic data:")
    print("   python scripts/generate_synthetic_data.py --num-samples 1000")
    print("\n4. Train a model:")
    print("   python scripts/train.py data=synthetic model=wav2vec2")
    print("\n5. Launch interactive demo:")
    print("   streamlit run demo/streamlit_app.py")
    print("   python demo/gradio_app.py")
    
    print("\n🔧 FEATURES:")
    print("• Modern ASR architectures (Wav2Vec2, Conformer)")
    print("• Transfer learning for low-resource languages")
    print("• Comprehensive evaluation metrics (WER, CER, confidence)")
    print("• Data augmentation (SpecAugment, speed/pitch perturbation)")
    print("• Privacy-preserving design with sanitization")
    print("• Interactive demos (Streamlit, Gradio)")
    print("• Synthetic data generation for testing")
    print("• CI/CD pipeline with GitHub Actions")
    
    print("\n⚠️  PRIVACY DISCLAIMER:")
    print("This project is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.")
    print("DO NOT USE for biometric identification, voice cloning, or")
    print("any malicious purposes. Always respect privacy rights and")
    print("data protection regulations.")
    
    print("\n📊 EVALUATION METRICS:")
    print("• Word Error Rate (WER) - Primary ASR metric")
    print("• Character Error Rate (CER) - Character-level accuracy")
    print("• Token Accuracy - Token-level precision")
    print("• Confidence Calibration - Reliability assessment")
    print("• Real-Time Factor (RTF) - Processing speed")
    print("• Throughput - Samples per second")
    
    print("\n🎯 USE CASES:")
    print("• Research in low-resource speech recognition")
    print("• Educational demonstrations of ASR techniques")
    print("• Transfer learning experiments")
    print("• Model architecture comparisons")
    print("• Evaluation metric benchmarking")
    
    print("\n" + "="*60)


def main():
    """Main function."""
    print("Low-Resource Speech Recognition Project Validation")
    print("=" * 50)
    
    # Validate project structure
    if not validate_project():
        logger.error("Project validation failed!")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        logger.error("Import tests failed!")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        logger.error("Functionality tests failed!")
        sys.exit(1)
    
    # Print summary
    print_project_summary()
    
    logger.info("Project validation completed successfully!")
    logger.info("The low-resource speech recognition project is ready to use!")


if __name__ == "__main__":
    main()
