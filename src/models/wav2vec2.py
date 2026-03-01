"""Wav2Vec2-based ASR model implementation."""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config

from ..utils.device import DeviceManager

logger = logging.getLogger(__name__)


class Wav2Vec2ASR(nn.Module):
    """Wav2Vec2-based Automatic Speech Recognition model."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h",
        vocab_size: int = 32,
        freeze_feature_extractor: bool = False,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        feat_proj_dropout: float = 0.0,
        layerdrop: float = 0.1,
        ctc_loss_reduction: str = "mean",
        pad_token_id: int = 0,
        ctc_zero_infinity: bool = False,
        device: Union[str, torch.device] = "auto",
    ):
        """Initialize Wav2Vec2 ASR model.
        
        Args:
            model_name: Pre-trained model name
            vocab_size: Vocabulary size
            freeze_feature_extractor: Whether to freeze feature extractor
            attention_dropout: Attention dropout rate
            hidden_dropout: Hidden dropout rate
            feat_proj_dropout: Feature projection dropout rate
            layerdrop: Layer dropout rate
            ctc_loss_reduction: CTC loss reduction method
            pad_token_id: Padding token ID
            ctc_zero_infinity: Whether to zero infinity in CTC loss
            device: Device to use
        """
        super().__init__()
        
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.device_manager = DeviceManager(device)
        
        # Load pre-trained model
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            feat_proj_dropout=feat_proj_dropout,
            layerdrop=layerdrop,
            ctc_loss_reduction=ctc_loss_reduction,
            pad_token_id=pad_token_id,
            ctc_zero_infinity=ctc_zero_infinity,
        )
        
        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Freeze feature extractor if requested
        if freeze_feature_extractor:
            self.model.freeze_feature_extractor()
            logger.info("Frozen feature extractor")
        
        # Move to device
        self.to(self.device_manager.device)
        
        logger.info(f"Initialized Wav2Vec2 ASR model: {model_name}")
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"Device: {self.device_manager.device}")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_values: Input audio features
            attention_mask: Attention mask
            labels: Target labels for training
            
        Returns:
            Dict containing logits and loss (if labels provided)
        """
        # Move inputs to device
        input_values = self.device_manager.to_device(input_values)
        if attention_mask is not None:
            attention_mask = self.device_manager.to_device(attention_mask)
        if labels is not None:
            labels = self.device_manager.to_device(labels)
        
        # Forward pass
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
        }
    
    def transcribe(
        self,
        audio: Union[torch.Tensor, str],
        sample_rate: int = 16000,
        return_confidence: bool = False,
    ) -> Union[str, tuple]:
        """Transcribe audio to text.
        
        Args:
            audio: Audio tensor or file path
            sample_rate: Sample rate of audio
            return_confidence: Whether to return confidence scores
            
        Returns:
            Transcription text or (text, confidence) tuple
        """
        self.eval()
        
        with torch.no_grad():
            # Load audio if path provided
            if isinstance(audio, str):
                import torchaudio
                waveform, sr = torchaudio.load(audio)
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                audio = waveform.squeeze(0)
            
            # Process audio
            inputs = self.processor(
                audio.numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = self.device_manager.to_device(inputs.input_values)
            attention_mask = self.device_manager.to_device(inputs.attention_mask)
            
            # Get logits
            outputs = self.forward(input_values, attention_mask)
            logits = outputs["logits"]
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
            
            if return_confidence:
                # Calculate confidence as max probability
                probs = torch.softmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0].mean().item()
                return transcription, confidence
            
            return transcription
    
    def fine_tune(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        output_dir: str = "./checkpoints",
    ):
        """Fine-tune the model on a dataset.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            save_steps: Saving frequency
            eval_steps: Evaluation frequency
            output_dir: Output directory for checkpoints
        """
        from transformers import TrainingArguments, Trainer
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save the model and processor.
        
        Args:
            save_directory: Directory to save to
        """
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "Wav2Vec2ASR":
        """Load pre-trained model.
        
        Args:
            model_name: Pre-trained model name
            **kwargs: Additional arguments
            
        Returns:
            Wav2Vec2ASR: Loaded model
        """
        return cls(model_name=model_name, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dict containing model information
        """
        from ..utils import count_parameters
        
        param_counts = count_parameters(self.model)
        
        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "device": str(self.device_manager.device),
            "parameters": param_counts,
            "feature_extractor_frozen": not self.model.feature_extractor.training,
        }
