"""Training utilities for speech recognition models."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer

from ..utils.device import DeviceManager
from ..utils import set_seed, PrivacyLogger

logger = logging.getLogger(__name__)


class ASRTrainer:
    """Trainer for ASR models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        device: Union[str, torch.device] = "auto",
        privacy_mode: bool = True,
    ):
        """Initialize ASR trainer.
        
        Args:
            model: ASR model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device to use for training
            privacy_mode: Whether to enable privacy protection
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device_manager = DeviceManager(device)
        self.privacy_mode = privacy_mode
        
        # Set up privacy logger
        if privacy_mode:
            self.logger = PrivacyLogger(logger)
        else:
            self.logger = logger
        
        # Move model to device
        self.model.to(self.device_manager.device)
        
        self.logger.info(f"Initialized ASR trainer on device: {self.device_manager.device}")
    
    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        output_dir: str = "./checkpoints",
        seed: int = 42,
        **kwargs
    ):
        """Train the model.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            save_steps: Saving frequency
            eval_steps: Evaluation frequency
            output_dir: Output directory for checkpoints
            seed: Random seed
            **kwargs: Additional training arguments
        """
        # Set seed for reproducibility
        set_seed(seed)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if self.val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="eval_loss" if self.val_dataset else None,
            greater_is_better=False if self.val_dataset else None,
            report_to="tensorboard",
            logging_dir=str(output_path / "logs"),
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            **kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=getattr(self.model, 'processor', None),
        )
        
        # Train
        self.logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        if hasattr(self.model, 'processor'):
            self.model.processor.save_pretrained(str(output_path))
        
        self.logger.info(f"Training completed. Model saved to {output_path}")
        
        return trainer


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: Whether to minimize or maximize the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early.
        
        Args:
            metrics: Current metrics
            
        Returns:
            bool: True if training should stop
        """
        if self.monitor not in metrics:
            logger.warning(f"Metric {self.monitor} not found in metrics")
            return False
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class CheckpointManager:
    """Checkpoint management utility."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            monitor: Metric to monitor
            mode: Whether to minimize or maximize the metric
            save_top_k: Number of best models to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        
        self.best_score = None
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
            filename: Custom filename
            
        Returns:
            str: Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update best score if applicable
        if self.save_best and self.monitor in metrics:
            current_score = metrics[self.monitor]
            
            if self.best_score is None or self._is_improvement(current_score, self.best_score):
                self.best_score = current_score
                
                # Save best model
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint_data, best_path)
                
                logger.info(f"New best model saved: {self.monitor} = {current_score:.4f}")
        
        # Manage top-k checkpoints
        self._manage_top_k(checkpoint_path, metrics)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            model: Model to load checkpoint into
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dict containing checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device_manager.device)
        
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint_data
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == "min":
            return current < best
        else:
            return current > best
    
    def _manage_top_k(self, checkpoint_path: Path, metrics: Dict[str, float]):
        """Manage top-k checkpoints."""
        if self.save_top_k <= 0:
            return
        
        # Add current checkpoint
        self.checkpoints.append((checkpoint_path, metrics.get(self.monitor, float('inf'))))
        
        # Sort by metric value
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda x: x[1], reverse=reverse)
        
        # Keep only top-k
        if len(self.checkpoints) > self.save_top_k:
            # Remove excess checkpoints
            for path, _ in self.checkpoints[self.save_top_k:]:
                if path.exists():
                    path.unlink()
            
            self.checkpoints = self.checkpoints[:self.save_top_k]


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer
        **kwargs: Additional optimizer arguments
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    scheduler_type: str = "linear",
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Scheduler
    """
    if scheduler_type.lower() == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_training_steps // 3,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
