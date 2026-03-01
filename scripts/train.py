#!/usr/bin/env python3
"""Training script for low-resource speech recognition."""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Wav2Vec2ASR, ConformerASR
from src.data import AudioDataset, SyntheticDataset, create_data_splits
from src.train import ASRTrainer, CheckpointManager, EarlyStopping
from src.utils import setup_logging, set_seed, PrivacyLogger
from src.metrics import ASRMetrics, evaluate_model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Set up logging
    setup_logging(cfg.logging.level, cfg.logging.file)
    logger = logging.getLogger(__name__)
    
    # Set up privacy logger if enabled
    if cfg.privacy.log_sanitization:
        logger = PrivacyLogger(logger)
    
    logger.info("Starting training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
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
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = create_data_splits(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=cfg.seed
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
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = create_data_splits(
            dataset,
            train_ratio=cfg.data.splits.train,
            val_ratio=cfg.data.splits.validation,
            test_ratio=cfg.data.splits.test,
            stratify_by=cfg.data.splits.stratify_by,
            random_state=cfg.seed
        )
    
    logger.info(f"Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
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
    
    logger.info(f"Model initialized: {model.get_model_info()}")
    
    # Initialize trainer
    trainer = ASRTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=cfg.device,
        privacy_mode=cfg.privacy.log_sanitization,
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoint_dir,
        save_best=cfg.training.checkpointing.save_best,
        monitor=cfg.training.checkpointing.monitor,
        mode=cfg.training.checkpointing.mode,
        save_top_k=cfg.training.checkpointing.save_top_k,
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        monitor=cfg.training.early_stopping.monitor,
        mode=cfg.training.early_stopping.mode,
    ) if cfg.training.early_stopping.enabled else None
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        num_epochs=cfg.training.epochs,
        learning_rate=cfg.training.optimizer.lr,
        batch_size=cfg.data.dataloader.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.scheduler.num_warmup_steps,
        logging_steps=cfg.training.logging.log_every_n_steps,
        save_steps=cfg.training.checkpointing.save_every_n_epochs * len(train_dataset) // cfg.data.dataloader.batch_size,
        eval_steps=cfg.training.validation.val_check_interval * len(train_dataset) // cfg.data.dataloader.batch_size,
        output_dir=str(output_dir),
        seed=cfg.seed,
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = ASRMetrics()
    test_results = evaluate_model(
        model=model,
        dataset=test_dataset,
        metrics=metrics,
        device=model.device_manager.device,
        batch_size=cfg.data.dataloader.batch_size,
    )
    
    # Log results
    logger.info("Test Results:")
    logger.info(f"WER: {test_results['asr_metrics']['wer']:.4f}")
    logger.info(f"CER: {test_results['asr_metrics']['cer']:.4f}")
    logger.info(f"Token Accuracy: {test_results['asr_metrics']['token_accuracy']:.4f}")
    logger.info(f"RTF: {test_results['performance_metrics']['real_time_factor']:.4f}")
    
    # Save results
    import json
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info(f"Training completed. Results saved to {results_path}")


if __name__ == "__main__":
    main()
