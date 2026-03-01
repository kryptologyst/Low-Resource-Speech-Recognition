"""Device utilities for handling different compute backends."""

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and memory optimization."""
    
    def __init__(self, device: Union[str, torch.device] = "auto"):
        """Initialize device manager.
        
        Args:
            device: Device preference ("auto", "cuda", "mps", "cpu")
        """
        self.device = self._get_device(device)
        self._setup_device()
    
    def _get_device(self, device: Union[str, torch.device]) -> torch.device:
        """Get the appropriate device for computation.
        
        Args:
            device: Device preference
            
        Returns:
            torch.device: The selected device
        """
        if isinstance(device, torch.device):
            return device
        
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _setup_device(self) -> None:
        """Set up device-specific configurations."""
        if self.device.type == "cuda":
            self._setup_cuda()
        elif self.device.type == "mps":
            self._setup_mps()
        else:
            self._setup_cpu()
    
    def _setup_cuda(self) -> None:
        """Set up CUDA-specific configurations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            raise RuntimeError("CUDA requested but not available")
    
    def _setup_mps(self) -> None:
        """Set up MPS-specific configurations."""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple Silicon MPS device")
        else:
            raise RuntimeError("MPS requested but not available")
    
    def _setup_cpu(self) -> None:
        """Set up CPU-specific configurations."""
        logger.info("Using CPU device")
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the managed device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            torch.Tensor: Tensor moved to device
        """
        return tensor.to(self.device)
    
    def clear_cache(self) -> None:
        """Clear device memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
    
    def get_memory_info(self) -> dict:
        """Get memory information for the device.
        
        Returns:
            dict: Memory information
        """
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9,
                "max_allocated": torch.cuda.max_memory_allocated() / 1e9,
            }
        elif self.device.type == "mps":
            return {
                "allocated": torch.mps.current_allocated_memory() / 1e9,
                "reserved": torch.mps.driver_allocated_memory() / 1e9,
            }
        else:
            import psutil
            return {
                "available": psutil.virtual_memory().available / 1e9,
                "used": psutil.virtual_memory().used / 1e9,
                "total": psutil.virtual_memory().total / 1e9,
            }


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    max_batch_size: int = 64,
    safety_factor: float = 0.8
) -> int:
    """Find optimal batch size for the given model and device.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Target device
        max_batch_size: Maximum batch size to test
        safety_factor: Safety factor for memory usage
        
    Returns:
        int: Optimal batch size
    """
    model = model.to(device)
    model.eval()
    
    optimal_batch_size = 1
    
    for batch_size in range(1, max_batch_size + 1):
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            optimal_batch_size = batch_size
            
            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            else:
                raise e
    
    # Apply safety factor
    optimal_batch_size = max(1, int(optimal_batch_size * safety_factor))
    
    logger.info(f"Optimal batch size for device {device}: {optimal_batch_size}")
    
    return optimal_batch_size
