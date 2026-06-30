"""
Training entrypoint for Siamese Object Tracker v2.

Loads configuration from a YAML file (default: config.yaml) and
CLI arguments can override any YAML value.

Usage:
    python train.py                              # uses default config.yaml
    python train.py --config myconfig.yaml       # custom config
    python train.py --max_epochs 5 --lr 5e-4     # override specific values
"""

import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.download_utils import download_and_extract_otb_sequence, download_teacher_weights
from data.datamodule import SiameseDataModule
from models.lightning_tracker import SiamTrackerLightning
from utils.pruning import SiamPruningCallback


def load_config(config_path):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    return cfg or {}


def deep_update(base, override):
    """Recursively update base dict with override dict."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_nested(d, *keys, default=None):
    """Safely get a value from a nested dict."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def main():
    # ------------------------------------------------------------------
    # Parse CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train Siamese Object Tracker v2 (MobileViT-XS + Linear Attention + Temporal Memory)")

    # Config file
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")

    # Override-able training parameters
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument("--train_seqs", nargs="+", default=None)
    parser.add_argument("--val_seqs", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Number of template frames for temporal memory")
    parser.add_argument("--enable_qat", action="store_true", default=None,
                        help="Enable Quantization-Aware Training")
    parser.add_argument("--precision", type=str, default=None,
                        help="Training precision: '32', '16-mixed', 'bf16-mixed'")
    parser.add_argument("--alpha_focal", type=float, default=None,
                        help="Focal loss alpha")
    parser.add_argument("--lambda_reg", type=float, default=None,
                        help="Regression loss weight")
    parser.add_argument("--beta_kd", type=float, default=None,
                        help="Knowledge distillation loss weight")
    parser.add_argument("--pruning_amount", type=float, default=None)
    parser.add_argument("--limit_train_batches", default=None)
    parser.add_argument("--limit_val_batches", default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load YAML config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # Resolve values: CLI args override YAML config
    data_dir = args.data_dir or get_nested(cfg, 'data', 'data_dir', default='data')
    weights_dir = args.weights_dir or get_nested(cfg, 'checkpoint', 'dirpath', default='weights')
    val_data_dir = get_nested(cfg, 'data', 'val_data_dir', default=None)
    
    # Auto-adjust path if cwd is 'src' but dataset is in the project root
    if not os.path.exists(data_dir):
        parent_data_dir = os.path.join("..", data_dir)
        if os.path.exists(parent_data_dir):
            data_dir = parent_data_dir

    if val_data_dir is not None and not os.path.exists(val_data_dir):
        parent_val_dir = os.path.join("..", val_data_dir)
        if os.path.exists(parent_val_dir):
            val_data_dir = parent_val_dir

    if val_data_dir is None:
        # Backward compatibility: split single data_dir dynamically if list.txt exists
        list_file = os.path.join(data_dir, "list.txt")
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                all_seqs = [line.strip() for line in f if line.strip()]
            all_seqs.sort()
            num_train = int(len(all_seqs) * 0.8)
            train_seqs = all_seqs[:num_train]
            val_seqs = all_seqs[num_train:]
            print(f"Detected sequence list file in {data_dir}. Dynamically split {len(all_seqs)} sequences into {len(train_seqs)} train and {len(val_seqs)} validation sequences (80:20).")
        else:
            train_seqs = args.train_seqs or get_nested(cfg, 'data', 'train_seqs', default=["Bolt", "Football"])
            val_seqs = args.val_seqs or get_nested(cfg, 'data', 'val_seqs', default=["Car1"])
        val_data_dir = data_dir
    else:
        # Load train sequences from data_dir
        train_list_file = os.path.join(data_dir, "list.txt")
        if os.path.exists(train_list_file):
            with open(train_list_file, 'r') as f:
                train_seqs = [line.strip() for line in f if line.strip()]
            train_seqs.sort()
        else:
            train_seqs = args.train_seqs or get_nested(cfg, 'data', 'train_seqs', default=["Bolt", "Football"])
            
        # Load validation sequences from val_data_dir
        val_list_file = os.path.join(val_data_dir, "list.txt")
        if os.path.exists(val_list_file):
            with open(val_list_file, 'r') as f:
                val_seqs = [line.strip() for line in f if line.strip()]
            val_seqs.sort()
        else:
            val_seqs = args.val_seqs or get_nested(cfg, 'data', 'val_seqs', default=["Car1"])
            
        print(f"Loaded {len(train_seqs)} train sequences from {data_dir} and {len(val_seqs)} validation sequences from {val_data_dir}.")
    batch_size = args.batch_size or get_nested(cfg, 'data', 'batch_size', default=8)
    max_epochs = args.max_epochs or get_nested(cfg, 'training', 'max_epochs', default=50)
    lr = args.lr or get_nested(cfg, 'optimizer', 'lr', default=1e-3)
    seq_len = args.seq_len or get_nested(cfg, 'data', 'seq_len', default=4)
    enable_qat = args.enable_qat if args.enable_qat is not None else get_nested(cfg, 'qat', 'enabled', default=False)
    precision = args.precision or get_nested(cfg, 'training', 'precision', default='16-mixed')
    alpha_focal = args.alpha_focal or get_nested(cfg, 'loss', 'focal', 'alpha', default=0.25)
    gamma_focal = get_nested(cfg, 'loss', 'focal', 'gamma', default=2.0)
    lambda_reg = args.lambda_reg or get_nested(cfg, 'loss', 'giou', 'weight', default=1.0)
    beta_kd = args.beta_kd or get_nested(cfg, 'loss', 'distillation', 'weight', default=1.0)
    pruning_amount = args.pruning_amount or get_nested(cfg, 'pruning', 'amount', default=0.3)
    pruning_enabled = get_nested(cfg, 'pruning', 'enabled', default=True)
    num_workers = args.num_workers if args.num_workers is not None else get_nested(cfg, 'data', 'num_workers', default=0)

    # Training control
    limit_train = args.limit_train_batches or get_nested(cfg, 'training', 'limit_train_batches', default=1.0)
    limit_val = args.limit_val_batches or get_nested(cfg, 'training', 'limit_val_batches', default=1.0)
    gradient_clip = args.gradient_clip_val or get_nested(cfg, 'training', 'gradient_clip_val', default=1.0)
    log_every_n_steps = get_nested(cfg, 'training', 'log_every_n_steps', default=5)

    # Augmentation
    scale_jitter = get_nested(cfg, 'augmentation', 'scale_jitter', default=0.1)
    shift_jitter = get_nested(cfg, 'augmentation', 'shift_jitter', default=16)
    max_search_gap = get_nested(cfg, 'data', 'max_search_gap', default=50)

    # Logging
    tb_dir = get_nested(cfg, 'logging', 'tensorboard_dir', default='lightning_logs')
    log_images_every = get_nested(cfg, 'logging', 'log_images_every_n_steps', default=1)
    num_vis_samples = get_nested(cfg, 'logging', 'num_vis_samples', default=4)

    # Checkpoint
    ckpt_filename = get_nested(cfg, 'checkpoint', 'filename', default='best_siam_tracker_v2')
    ckpt_monitor = get_nested(cfg, 'checkpoint', 'monitor', default='val_loss')
    ckpt_mode = get_nested(cfg, 'checkpoint', 'mode', default='min')

    # Backbone config
    backbone_cfg = {
        'stem_channels': get_nested(cfg, 'model', 'backbone', 'stem_channels', default=16),
        'stage_configs': get_nested(cfg, 'model', 'backbone', 'stages', default=None),
        'transformer_depth': get_nested(cfg, 'model', 'backbone', 'transformer', 'depth', default=2),
        'heads_stage4': get_nested(cfg, 'model', 'backbone', 'transformer', 'heads_stage4', default=2),
        'heads_stage5': get_nested(cfg, 'model', 'backbone', 'transformer', 'heads_stage5', default=4),
        'mlp_ratio': get_nested(cfg, 'model', 'backbone', 'transformer', 'mlp_ratio', default=2.0),
        'dropout': get_nested(cfg, 'model', 'backbone', 'transformer', 'dropout', default=0.0),
    }

    # Memory config
    memory_cfg = {
        'cache_size': get_nested(cfg, 'model', 'temporal_memory', 'cache_size', default=4),
        'state_dim': get_nested(cfg, 'model', 'temporal_memory', 'state_dim', default=16),
    }

    # Head config
    head_cfg = {
        'hidden_channels': get_nested(cfg, 'model', 'head', 'hidden_channels', default=64),
        'num_cls_convs': get_nested(cfg, 'model', 'head', 'num_cls_convs', default=1),
        'num_reg_convs': get_nested(cfg, 'model', 'head', 'num_reg_convs', default=1),
    }

    # QAT config
    qconfig_backend = get_nested(cfg, 'qat', 'qconfig_backend', default='fbgemm')

    # Optimizer config
    weight_decay = get_nested(cfg, 'optimizer', 'weight_decay', default=1e-4)
    betas = tuple(get_nested(cfg, 'optimizer', 'betas', default=[0.9, 0.999]))
    eta_min = get_nested(cfg, 'scheduler', 'eta_min', default=1e-6)

    # Teacher config
    teacher_layer_channels = get_nested(cfg, 'teacher', 'layer_channels', default=[512, 1024, 2048])

    def parse_limit(val):
        if val is None:
            return 1.0
        if isinstance(val, (int, float)):
            return val
        val_str = str(val).strip()
        if '.' in val_str:
            return float(val_str)
        try:
            return int(val_str)
        except ValueError:
            return float(val_str)

    limit_train = parse_limit(limit_train)
    limit_val = parse_limit(limit_val)

    # ------------------------------------------------------------------
    # Print Configuration Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Siamese Object Tracker v2 — Training Configuration")
    print("=" * 60)
    print(f"  Config file:        {args.config}")
    print(f"  Data dir:           {data_dir}")
    print(f"  Train sequences:    {train_seqs}")
    print(f"  Val sequences:      {val_seqs}")
    print(f"  Seq len (T):        {seq_len}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Max epochs:         {max_epochs}")
    print(f"  Learning rate:      {lr}")
    print(f"  Precision:          {precision}")
    print(f"  QAT enabled:        {enable_qat}")
    print(f"  Loss weights:       cls(focal alpha={alpha_focal}), reg(lambda={lambda_reg}), kd(beta={beta_kd})")
    print(f"  Pruning:            {pruning_enabled} (amount={pruning_amount})")
    print(f"  TensorBoard dir:    {tb_dir}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Download data & teacher weights
    # ------------------------------------------------------------------
    print("=== Step 1: Downloading Datasets & Teacher Weights ===")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Only download if we are NOT using the local list.txt dataset
    if not os.path.exists(os.path.join(data_dir, "list.txt")):
        for seq in train_seqs:
            download_and_extract_otb_sequence(seq, data_dir)
    else:
        print(f"Detected local sequence list in {data_dir}. Skipping OTB sequence download.")

    teacher_weights_path = download_teacher_weights(weights_dir)

    # ------------------------------------------------------------------
    # Step 2: Setup DataModule
    # ------------------------------------------------------------------
    print("\n=== Step 2: Setting up Data Pipeline ===")
    datamodule = SiameseDataModule(
        data_dir=data_dir,
        val_data_dir=val_data_dir,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        seq_len=seq_len,
        max_search_gap=max_search_gap,
        batch_size=batch_size,
        num_workers=num_workers,
        scale_jitter=scale_jitter,
        shift_jitter=shift_jitter,
    )

    # ------------------------------------------------------------------
    # Step 3: Setup Model
    # ------------------------------------------------------------------
    print("\n=== Step 3: Initializing SiamTrackerLightning Model ===")
    model = SiamTrackerLightning(
        backbone_cfg=backbone_cfg,
        memory_cfg=memory_cfg,
        head_cfg=head_cfg,
        teacher_weights_path=teacher_weights_path,
        teacher_layer_channels=teacher_layer_channels,
        alpha_focal=alpha_focal,
        gamma_focal=gamma_focal,
        lambda_reg=lambda_reg,
        beta_kd=beta_kd,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eta_min=eta_min,
        enable_qat=enable_qat,
        qconfig_backend=qconfig_backend,
        log_images_every_n_steps=log_images_every,
        num_vis_samples=num_vis_samples,
    )

    # ------------------------------------------------------------------
    # Step 4: Configure Callbacks & Logger
    # ------------------------------------------------------------------
    print("\n=== Step 4: Setting up Callbacks & Trainer ===")

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=1,
        verbose=True,
    )

    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval='epoch')]

    if pruning_enabled:
        callbacks.append(SiamPruningCallback(pruning_amount=pruning_amount))

    logger = TensorBoardLogger(save_dir=".", name=tb_dir, default_hp_metric=False)

    # Accelerator
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {accelerator.upper()}")

    # ------------------------------------------------------------------
    # Step 5: Train
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=1.0,
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,
        gradient_clip_val=gradient_clip,
        precision=precision if accelerator == "gpu" else 32,
    )

    print("\n=== Step 5: Launching Trainer.fit() ===")
    trainer.fit(model, datamodule=datamodule)

    print("\n=== Training Completed Successfully! ===")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
