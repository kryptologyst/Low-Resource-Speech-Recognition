"""Core utilities for the low-resource speech recognition project."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for privacy protection.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace potentially identifying information
    import hashlib
    import re
    
    # Extract extension
    name, ext = os.path.splitext(filename)
    
    # Remove common identifying patterns
    name = re.sub(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', 'DATE', name)  # Dates
    name = re.sub(r'[0-9]{3}-[0-9]{3}-[0-9]{4}', 'PHONE', name)  # Phone numbers
    name = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'EMAIL', name)  # Emails
    
    # Hash the remaining name for privacy
    hashed_name = hashlib.md5(name.encode()).hexdigest()[:8]
    
    return f"{hashed_name}{ext}"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict[str, int]: Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> str:
    """Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        str: Human-readable file size
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return "File not found"
    
    size_bytes = file_path.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"


class PrivacyLogger:
    """Logger that sanitizes potentially sensitive information."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize log message for privacy."""
        import re
        
        # Remove common PII patterns
        message = re.sub(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', '[DATE]', message)
        message = re.sub(r'[0-9]{3}-[0-9]{3}-[0-9]{4}', '[PHONE]', message)
        message = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', message)
        
        return message
    
    def debug(self, message: str) -> None:
        self.logger.debug(self._sanitize_message(message))
    
    def info(self, message: str) -> None:
        self.logger.info(self._sanitize_message(message))
    
    def warning(self, message: str) -> None:
        self.logger.warning(self._sanitize_message(message))
    
    def error(self, message: str) -> None:
        self.logger.error(self._sanitize_message(message))
    
    def critical(self, message: str) -> None:
        self.logger.critical(self._sanitize_message(message))
