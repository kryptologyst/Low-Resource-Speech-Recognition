"""Evaluation metrics for speech recognition."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from jiwer import wer, cer, compute_measures

logger = logging.getLogger(__name__)


class ASRMetrics:
    """ASR evaluation metrics."""
    
    def __init__(self, vocab: Optional[List[str]] = None):
        """Initialize ASR metrics.
        
        Args:
            vocab: Vocabulary list for token-level metrics
        """
        self.vocab = vocab or []
        self.vocab_set = set(self.vocab)
    
    def word_error_rate(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Word Error Rate (WER).
        
        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            
        Returns:
            float: Word Error Rate
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        return wer(references, predictions)
    
    def character_error_rate(self, predictions: List[str], references: List[str]) -> float:
        """Calculate Character Error Rate (CER).
        
        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            
        Returns:
            float: Character Error Rate
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        return cer(references, predictions)
    
    def token_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate token-level accuracy.
        
        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            
        Returns:
            float: Token accuracy
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            total_tokens += len(ref_tokens)
            
            # Count correct tokens
            for i, ref_token in enumerate(ref_tokens):
                if i < len(pred_tokens) and pred_tokens[i] == ref_token:
                    correct_tokens += 1
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute all ASR metrics.
        
        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            
        Returns:
            Dict containing all metrics
        """
        # Compute jiwer measures
        measures = compute_measures(references, predictions)
        
        # Calculate additional metrics
        wer_score = self.word_error_rate(predictions, references)
        cer_score = self.character_error_rate(predictions, references)
        token_acc = self.token_accuracy(predictions, references)
        
        return {
            "wer": wer_score,
            "cer": cer_score,
            "token_accuracy": token_acc,
            "hits": measures["hits"],
            "substitutions": measures["substitutions"],
            "deletions": measures["deletions"],
            "insertions": measures["insertions"],
            "substitutions_rate": measures["substitutions"] / measures["truth_words"] if measures["truth_words"] > 0 else 0,
            "deletions_rate": measures["deletions"] / measures["truth_words"] if measures["truth_words"] > 0 else 0,
            "insertions_rate": measures["insertions"] / measures["truth_words"] if measures["truth_words"] > 0 else 0,
        }


class ConfidenceCalibration:
    """Confidence calibration metrics."""
    
    def __init__(self, num_bins: int = 10):
        """Initialize confidence calibration.
        
        Args:
            num_bins: Number of bins for calibration
        """
        self.num_bins = num_bins
    
    def expected_calibration_error(
        self,
        confidences: List[float],
        accuracies: List[bool],
    ) -> float:
        """Calculate Expected Calibration Error (ECE).
        
        Args:
            confidences: Confidence scores
            accuracies: Binary accuracy values
            
        Returns:
            float: Expected Calibration Error
        """
        if len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have the same length")
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def maximum_calibration_error(
        self,
        confidences: List[float],
        accuracies: List[bool],
    ) -> float:
        """Calculate Maximum Calibration Error (MCE).
        
        Args:
            confidences: Confidence scores
            accuracies: Binary accuracy values
            
        Returns:
            float: Maximum Calibration Error
        """
        if len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have the same length")
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Update MCE
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce


class PerformanceMetrics:
    """Performance metrics for ASR systems."""
    
    def __init__(self):
        """Initialize performance metrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_audio_duration = 0.0
        self.total_inference_time = 0.0
        self.total_samples = 0
        self.memory_usage = []
    
    def add_sample(
        self,
        audio_duration: float,
        inference_time: float,
        memory_usage: Optional[float] = None,
    ):
        """Add a sample for performance calculation.
        
        Args:
            audio_duration: Duration of audio in seconds
            inference_time: Inference time in seconds
            memory_usage: Memory usage in GB (optional)
        """
        self.total_audio_duration += audio_duration
        self.total_inference_time += inference_time
        self.total_samples += 1
        
        if memory_usage is not None:
            self.memory_usage.append(memory_usage)
    
    def real_time_factor(self) -> float:
        """Calculate Real-Time Factor (RTF).
        
        Returns:
            float: Real-Time Factor
        """
        if self.total_audio_duration == 0:
            return 0.0
        
        return self.total_inference_time / self.total_audio_duration
    
    def throughput(self) -> float:
        """Calculate throughput (samples per second).
        
        Returns:
            float: Throughput
        """
        if self.total_inference_time == 0:
            return 0.0
        
        return self.total_samples / self.total_inference_time
    
    def average_memory_usage(self) -> Optional[float]:
        """Calculate average memory usage.
        
        Returns:
            float: Average memory usage in GB, or None if no data
        """
        if not self.memory_usage:
            return None
        
        return np.mean(self.memory_usage)
    
    def peak_memory_usage(self) -> Optional[float]:
        """Calculate peak memory usage.
        
        Returns:
            float: Peak memory usage in GB, or None if no data
        """
        if not self.memory_usage:
            return None
        
        return np.max(self.memory_usage)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all performance metrics.
        
        Returns:
            Dict containing all performance metrics
        """
        metrics = {
            "real_time_factor": self.real_time_factor(),
            "throughput": self.throughput(),
            "total_samples": self.total_samples,
            "total_audio_duration": self.total_audio_duration,
            "total_inference_time": self.total_inference_time,
        }
        
        avg_memory = self.average_memory_usage()
        if avg_memory is not None:
            metrics["average_memory_usage_gb"] = avg_memory
        
        peak_memory = self.peak_memory_usage()
        if peak_memory is not None:
            metrics["peak_memory_usage_gb"] = peak_memory
        
        return metrics


def evaluate_model(
    model,
    dataset,
    metrics: Optional[ASRMetrics] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate a model on a dataset.
    
    Args:
        model: ASR model to evaluate
        dataset: Dataset to evaluate on
        metrics: ASR metrics calculator
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        
    Returns:
        Dict containing evaluation results
    """
    if metrics is None:
        metrics = ASRMetrics()
    
    model.eval()
    
    predictions = []
    references = []
    confidences = []
    performance = PerformanceMetrics()
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Get reference text
            ref_text = sample["text"]
            references.append(ref_text)
            
            # Get features
            if "features" in sample:
                features = sample["features"].unsqueeze(0)
            elif "input_values" in sample:
                features = sample["input_values"].unsqueeze(0)
            else:
                raise ValueError("No features found in sample")
            
            # Move to device if specified
            if device is not None:
                features = features.to(device)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time is not None:
                start_time.record()
            
            # Transcribe
            if hasattr(model, 'transcribe'):
                result = model.transcribe(features, return_confidence=True)
                if isinstance(result, tuple):
                    pred_text, confidence = result
                    confidences.append(confidence)
                else:
                    pred_text = result
                    confidences.append(1.0)  # Default confidence
            else:
                # Fallback for models without transcribe method
                outputs = model(features)
                logits = outputs["logits"]
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_text = " ".join([str(id.item()) for id in predicted_ids[0]])
                confidences.append(1.0)
            
            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                inference_time = 0.0  # Fallback
            
            predictions.append(pred_text)
            
            # Add to performance metrics
            audio_duration = sample.get("duration", 0.0)
            performance.add_sample(audio_duration, inference_time)
    
    # Calculate metrics
    asr_metrics = metrics.compute_all_metrics(predictions, references)
    performance_metrics = performance.get_all_metrics()
    
    # Calculate confidence calibration
    calibration = ConfidenceCalibration()
    accuracies = [pred == ref for pred, ref in zip(predictions, references)]
    ece = calibration.expected_calibration_error(confidences, accuracies)
    mce = calibration.maximum_calibration_error(confidences, accuracies)
    
    return {
        "asr_metrics": asr_metrics,
        "performance_metrics": performance_metrics,
        "confidence_calibration": {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "average_confidence": np.mean(confidences),
        },
        "predictions": predictions,
        "references": references,
        "confidences": confidences,
    }
