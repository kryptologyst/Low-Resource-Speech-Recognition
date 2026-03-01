"""Conformer-based ASR model implementation."""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from ..utils.device import DeviceManager

logger = logging.getLogger(__name__)


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(
        self,
        input_dim: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ):
        """Initialize convolution module.
        
        Args:
            input_dim: Input dimension
            kernel_size: Convolution kernel size
            expansion_factor: Expansion factor
            dropout_p: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        
        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(
            input_dim,
            input_dim * expansion_factor,
            kernel_size=1,
        )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            input_dim * expansion_factor,
            input_dim * expansion_factor,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=input_dim * expansion_factor,
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(input_dim * expansion_factor)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(
            input_dim * expansion_factor,
            input_dim,
            kernel_size=1,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Transpose for convolution
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        # Pointwise convolution 1
        x = self.pointwise_conv1(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Batch normalization
        x = self.batch_norm(x)
        
        # Activation
        x = self.activation(x)
        
        # Pointwise convolution 2
        x = self.pointwise_conv2(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # [batch, seq_len, input_dim]
        
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer."""
    
    def __init__(
        self,
        input_dim: int,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ):
        """Initialize feed-forward module.
        
        Args:
            input_dim: Input dimension
            expansion_factor: Expansion factor
            dropout_p: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        
        # Linear layers
        self.linear1 = nn.Linear(input_dim, input_dim * expansion_factor)
        self.linear2 = nn.Linear(input_dim * expansion_factor, input_dim)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class ConformerBlock(nn.Module):
    """Conformer block."""
    
    def __init__(
        self,
        input_dim: int,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        """Initialize Conformer block.
        
        Args:
            input_dim: Input dimension
            num_attention_heads: Number of attention heads
            feed_forward_expansion_factor: Feed-forward expansion factor
            conv_expansion_factor: Convolution expansion factor
            feed_forward_dropout_p: Feed-forward dropout probability
            attention_dropout_p: Attention dropout probability
            conv_dropout_p: Convolution dropout probability
            conv_kernel_size: Convolution kernel size
            half_step_residual: Whether to use half-step residual
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.half_step_residual = half_step_residual
        
        # Feed-forward module 1
        self.feed_forward1 = FeedForwardModule(
            input_dim=input_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        
        # Multi-head self-attention
        self.self_attention = MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout_p,
            batch_first=True,
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        
        # Convolution module
        self.conv_module = ConvolutionModule(
            input_dim=input_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
        )
        
        # Feed-forward module 2
        self.feed_forward2 = FeedForwardModule(
            input_dim=input_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Output tensor
        """
        residual = x
        
        # Feed-forward module 1
        x = self.feed_forward1(x)
        if self.half_step_residual:
            x = x * 0.5
        x = residual + x
        x = self.layer_norm1(x)
        
        # Multi-head self-attention
        residual = x
        x, _ = self.self_attention(x, x, x, attn_mask=attention_mask)
        x = residual + x
        x = self.layer_norm2(x)
        
        # Convolution module
        residual = x
        x = self.conv_module(x)
        x = residual + x
        x = self.layer_norm3(x)
        
        # Feed-forward module 2
        residual = x
        x = self.feed_forward2(x)
        if self.half_step_residual:
            x = x * 0.5
        x = residual + x
        
        # Final layer normalization
        x = self.final_layer_norm(x)
        
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder."""
    
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int = 512,
        num_encoder_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        """Initialize Conformer encoder.
        
        Args:
            input_dim: Input feature dimension
            encoder_dim: Encoder dimension
            num_encoder_layers: Number of encoder layers
            num_attention_heads: Number of attention heads
            feed_forward_expansion_factor: Feed-forward expansion factor
            conv_expansion_factor: Convolution expansion factor
            input_dropout_p: Input dropout probability
            feed_forward_dropout_p: Feed-forward dropout probability
            attention_dropout_p: Attention dropout probability
            conv_dropout_p: Convolution dropout probability
            conv_kernel_size: Convolution kernel size
            half_step_residual: Whether to use half-step residual
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, encoder_dim)
        
        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout_p)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                input_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            )
            for _ in range(num_encoder_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Encoded tensor [batch, seq_len, encoder_dim]
        """
        # Input projection
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, attention_mask)
        
        return x


class ConformerASR(nn.Module):
    """Conformer-based Automatic Speech Recognition model."""
    
    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_encoder_layers: int = 17,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        vocab_size: int = 5000,
        blank_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 3,
        device: Union[str, torch.device] = "auto",
    ):
        """Initialize Conformer ASR model.
        
        Args:
            input_dim: Input feature dimension
            encoder_dim: Encoder dimension
            num_encoder_layers: Number of encoder layers
            num_attention_heads: Number of attention heads
            feed_forward_expansion_factor: Feed-forward expansion factor
            conv_expansion_factor: Convolution expansion factor
            input_dropout_p: Input dropout probability
            feed_forward_dropout_p: Feed-forward dropout probability
            attention_dropout_p: Attention dropout probability
            conv_dropout_p: Convolution dropout probability
            conv_kernel_size: Convolution kernel size
            half_step_residual: Whether to use half-step residual
            vocab_size: Vocabulary size
            blank_id: Blank token ID
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            pad_id: Padding token ID
            device: Device to use
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        
        self.device_manager = DeviceManager(device)
        
        # Conformer encoder
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_encoder_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        
        # CTC classifier
        self.ctc_classifier = nn.Linear(encoder_dim, vocab_size)
        
        # Move to device
        self.to(self.device_manager.device)
        
        logger.info(f"Initialized Conformer ASR model")
        logger.info(f"Input dim: {input_dim}, Encoder dim: {encoder_dim}")
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"Device: {self.device_manager.device}")
    
    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: Input features [batch, seq_len, input_dim]
            attention_mask: Attention mask
            labels: Target labels for training
            
        Returns:
            Dict containing logits and loss (if labels provided)
        """
        # Move inputs to device
        features = self.device_manager.to_device(features)
        if attention_mask is not None:
            attention_mask = self.device_manager.to_device(attention_mask)
        if labels is not None:
            labels = self.device_manager.to_device(labels)
        
        # Encode features
        encoded = self.encoder(features, attention_mask)
        
        # CTC classification
        logits = self.ctc_classifier(encoded)
        
        # Calculate CTC loss if labels provided
        loss = None
        if labels is not None:
            # CTC loss expects [seq_len, batch, vocab_size]
            logits = logits.transpose(0, 1)
            
            # Calculate CTC loss
            loss = F.ctc_loss(
                logits,
                labels,
                input_lengths=attention_mask.sum(dim=1) if attention_mask is not None else None,
                target_lengths=(labels != self.pad_id).sum(dim=1),
                blank=self.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
        
        return {
            "logits": logits.transpose(0, 1) if loss is not None else logits,
            "loss": loss,
            "encoded": encoded,
        }
    
    def transcribe(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
    ) -> Union[str, tuple]:
        """Transcribe features to text.
        
        Args:
            features: Input features
            attention_mask: Attention mask
            return_confidence: Whether to return confidence scores
            
        Returns:
            Transcription text or (text, confidence) tuple
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(features, attention_mask)
            logits = outputs["logits"]
            
            # Decode using greedy decoding
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Remove blank tokens
            predicted_ids = predicted_ids[predicted_ids != self.blank_id]
            
            # Convert to text (simplified - would need proper tokenizer)
            text = " ".join([str(id.item()) for id in predicted_ids])
            
            if return_confidence:
                # Calculate confidence as max probability
                probs = torch.softmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0].mean().item()
                return text, confidence
            
            return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dict containing model information
        """
        from ..utils import count_parameters
        
        param_counts = count_parameters(self)
        
        return {
            "model_type": "Conformer",
            "input_dim": self.input_dim,
            "encoder_dim": self.encoder_dim,
            "vocab_size": self.vocab_size,
            "device": str(self.device_manager.device),
            "parameters": param_counts,
        }
