#!/usr/bin/env python3
"""Evaluation script for low-resource speech recognition."""

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Wav2Vec2ASR, ConformerASR
from src.data import AudioDataset, SyntheticDataset
from src.metrics import ASRMetrics, evaluate_model
from src.utils import setup_logging, PrivacyLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Set up logging
    setup_logging(cfg.logging.level, cfg.logging.file)
    logger = logging.getLogger(__name__)
    
    # Set up privacy logger if enabled
    if cfg.privacy.log_sanitization:
        logger = PrivacyLogger(logger)
    
    logger.info("Starting evaluation...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    if cfg.data.dataset_name == "synthetic":
        # Create synthetic dataset for demonstration
        dataset = SyntheticDataset(
            num_samples=cfg.data.get("num_samples", 1000),
            sample_rate=cfg.data.audio.sample_rate,
            duration_range=cfg.data.get("duration_range", (1.0, 5.0)),
            vocab_size=cfg.data.get("vocab_size", 100),
            feature_type=cfg.data.features.feature_type,
            **cfg.data.features
        )
    else:
        # Load real dataset
        dataset = AudioDataset(
            data_dir=cfg.data.paths.data_dir,
            meta_file=cfg.data.paths.meta_file,
            audio_dir=cfg.data.paths.audio_dir,
            feature_type=cfg.data.features.feature_type,
            sample_rate=cfg.data.audio.sample_rate,
            max_duration=cfg.data.audio.max_duration,
            min_duration=cfg.data.audio.min_duration,
            normalize=cfg.data.audio.normalize,
            preemphasis=cfg.data.audio.preemphasis,
            trim_silence=cfg.data.audio.trim_silence,
            privacy_mode=cfg.privacy.anonymize_filenames,
            **cfg.data.features
        )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Load model
    logger.info("Loading model...")
    checkpoint_path = cfg.get("checkpoint_path", None)
    
    if checkpoint_path is None:
        # Load from config
        if cfg.model._target_.endswith("Wav2Vec2ASR"):
            model = Wav2Vec2ASR(
                model_name=cfg.model.architecture.model_name,
                vocab_size=cfg.model.architecture.vocab_size,
                freeze_feature_extractor=cfg.model.architecture.freeze_feature_extractor,
                attention_dropout=cfg.model.architecture.attention_dropout,
                hidden_dropout=cfg.model.architecture.hidden_dropout,
                feat_proj_dropout=cfg.model.architecture.feat_proj_dropout,
                layerdrop=cfg.model.architecture.layerdrop,
                ctc_loss_reduction=cfg.model.architecture.ctc_loss_reduction,
                pad_token_id=cfg.model.architecture.pad_token_id,
                ctc_zero_infinity=cfg.model.architecture.ctc_zero_infinity,
                device=cfg.device,
            )
        elif cfg.model._target_.endswith("ConformerASR"):
            model = ConformerASR(
                input_dim=cfg.model.architecture.input_dim,
                encoder_dim=cfg.model.architecture.encoder_dim,
                num_encoder_layers=cfg.model.architecture.num_encoder_layers,
                num_attention_heads=cfg.model.architecture.num_attention_heads,
                feed_forward_expansion_factor=cfg.model.architecture.feed_forward_expansion_factor,
                conv_expansion_factor=cfg.model.architecture.conv_expansion_factor,
                input_dropout_p=cfg.model.architecture.input_dropout_p,
                feed_forward_dropout_p=cfg.model.architecture.feed_forward_dropout_p,
                attention_dropout_p=cfg.model.architecture.attention_dropout_p,
                conv_dropout_p=cfg.model.architecture.conv_dropout_p,
                conv_kernel_size=cfg.model.architecture.conv_kernel_size,
                half_step_residual=cfg.model.architecture.half_step_residual,
                vocab_size=cfg.model.vocab_size,
                blank_id=cfg.model.blank_id,
                sos_id=cfg.model.sos_id,
                eos_id=cfg.model.eos_id,
                pad_id=cfg.model.pad_id,
                device=cfg.device,
            )
        else:
            raise ValueError(f"Unknown model type: {cfg.model._target_}")
    else:
        # Load from checkpoint
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Initialize model
        if cfg.model._target_.endswith("Wav2Vec2ASR"):
            model = Wav2Vec2ASR(
                model_name=cfg.model.architecture.model_name,
                vocab_size=cfg.model.architecture.vocab_size,
                device=cfg.device,
            )
        elif cfg.model._target_.endswith("ConformerASR"):
            model = ConformerASR(
                input_dim=cfg.model.architecture.input_dim,
                encoder_dim=cfg.model.architecture.encoder_dim,
                vocab_size=cfg.model.vocab_size,
                device=cfg.device,
            )
        else:
            raise ValueError(f"Unknown model type: {cfg.model._target_}")
        
        # Load state dict
        model.load_state_dict(checkpoint_data["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    logger.info(f"Model loaded: {model.get_model_info()}")
    
    # Initialize metrics
    metrics = ASRMetrics()
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = evaluate_model(
        model=model,
        dataset=dataset,
        metrics=metrics,
        device=model.device_manager.device,
        batch_size=cfg.data.dataloader.batch_size,
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    logger.info("ASR Metrics:")
    for metric, value in results["asr_metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nPerformance Metrics:")
    for metric, value in results["performance_metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nConfidence Calibration:")
    for metric, value in results["confidence_calibration"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    
    # Generate report if requested
    if cfg.get("generate_report", False):
        generate_report(results, output_dir)
    
    logger.info("Evaluation completed.")


def generate_report(results: dict, output_dir: Path) -> None:
    """Generate evaluation report.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Speech Recognition Evaluation Report", fontsize=16, fontweight="bold")
    
    # ASR Metrics
    ax1 = axes[0, 0]
    asr_metrics = results["asr_metrics"]
    metrics_names = ["WER", "CER", "Token Accuracy"]
    metrics_values = [asr_metrics["wer"], asr_metrics["cer"], asr_metrics["token_accuracy"]]
    
    bars = ax1.bar(metrics_names, metrics_values, color=["red", "orange", "green"])
    ax1.set_title("ASR Performance Metrics")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Error Analysis
    ax2 = axes[0, 1]
    error_types = ["Substitutions", "Deletions", "Insertions"]
    error_rates = [
        asr_metrics["substitutions_rate"],
        asr_metrics["deletions_rate"],
        asr_metrics["insertions_rate"]
    ]
    
    bars = ax2.bar(error_types, error_rates, color=["red", "blue", "purple"])
    ax2.set_title("Error Type Analysis")
    ax2.set_ylabel("Rate")
    
    # Add value labels on bars
    for bar, value in zip(bars, error_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Performance Metrics
    ax3 = axes[1, 0]
    perf_metrics = results["performance_metrics"]
    perf_names = ["RTF", "Throughput"]
    perf_values = [perf_metrics["real_time_factor"], perf_metrics["throughput"]]
    
    bars = ax3.bar(perf_names, perf_values, color=["blue", "green"])
    ax3.set_title("Performance Metrics")
    ax3.set_ylabel("Value")
    
    # Add value labels on bars
    for bar, value in zip(bars, perf_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Confidence Distribution
    ax4 = axes[1, 1]
    confidences = results["confidences"]
    ax4.hist(confidences, bins=20, alpha=0.7, color="purple", edgecolor="black")
    ax4.set_title("Confidence Score Distribution")
    ax4.set_xlabel("Confidence Score")
    ax4.set_ylabel("Frequency")
    ax4.axvline(np.mean(confidences), color="red", linestyle="--", 
                label=f"Mean: {np.mean(confidences):.3f}")
    ax4.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "evaluation_report.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create text report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Speech Recognition Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ASR Performance:\n")
        f.write(f"  Word Error Rate (WER): {asr_metrics['wer']:.4f}\n")
        f.write(f"  Character Error Rate (CER): {asr_metrics['cer']:.4f}\n")
        f.write(f"  Token Accuracy: {asr_metrics['token_accuracy']:.4f}\n\n")
        
        f.write("Error Analysis:\n")
        f.write(f"  Substitutions: {asr_metrics['substitutions']} ({asr_metrics['substitutions_rate']:.4f})\n")
        f.write(f"  Deletions: {asr_metrics['deletions']} ({asr_metrics['deletions_rate']:.4f})\n")
        f.write(f"  Insertions: {asr_metrics['insertions']} ({asr_metrics['insertions_rate']:.4f})\n\n")
        
        f.write("Performance:\n")
        f.write(f"  Real-Time Factor: {perf_metrics['real_time_factor']:.4f}\n")
        f.write(f"  Throughput: {perf_metrics['throughput']:.4f} samples/sec\n")
        f.write(f"  Total Samples: {perf_metrics['total_samples']}\n")
        f.write(f"  Total Audio Duration: {perf_metrics['total_audio_duration']:.2f} seconds\n")
        f.write(f"  Total Inference Time: {perf_metrics['total_inference_time']:.2f} seconds\n\n")
        
        f.write("Confidence Calibration:\n")
        calib = results["confidence_calibration"]
        f.write(f"  Expected Calibration Error: {calib['expected_calibration_error']:.4f}\n")
        f.write(f"  Maximum Calibration Error: {calib['maximum_calibration_error']:.4f}\n")
        f.write(f"  Average Confidence: {calib['average_confidence']:.4f}\n")
    
    logger.info(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
