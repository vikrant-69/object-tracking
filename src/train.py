import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.download_utils import download_and_extract_otb_sequence, download_teacher_weights
from data.datamodule import SiameseDataModule
from models.lightning_tracker import SiamTrackerLightning
from utils.pruning import SiamPruningCallback

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight Siamese Object Tracker")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to save datasets")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save checkpoints/weights")
    parser.add_argument("--train_seqs", nargs="+", default=["Bolt", "Football"], help="Sequences to train on")
    parser.add_argument("--val_seqs", nargs="+", default=["Car1"], help="Sequences to validate on")
    parser.add_argument("--batch_size", type=type(8), default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=type(10), default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--pruning_amount", type=float, default=0.3, help="Structured pruning fraction")
    parser.add_argument("--beta_kd", type=float, default=1.0, help="Weight for Knowledge Distillation loss")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for bounding box regression loss")
    parser.add_argument("--limit_train_batches", type=type(10), default=10, help="Limit train batches per epoch")
    parser.add_argument("--limit_val_batches", type=type(5), default=5, help="Limit val batches per epoch")
    args = parser.parse_args()

    # Step 1: Download training/validation sequences and teacher weights
    print("\n=== Step 1: Downloading Datasets & Teacher Weights ===")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    
    for seq in args.train_seqs + args.val_seqs:
        download_and_extract_otb_sequence(seq, args.data_dir)
        
    teacher_weights_path = download_teacher_weights(args.weights_dir)

    # Step 2: Setup DataModule
    print("\n=== Step 2: Setting up Data Pipeline ===")
    datamodule = SiameseDataModule(
        data_dir=args.data_dir,
        train_seqs=args.train_seqs,
        val_seqs=args.val_seqs,
        batch_size=args.batch_size,
        num_workers=0  # Use 0 workers on Windows for stability during testing
    )

    # Step 3: Setup Model
    print("\n=== Step 3: Initializing SiamTrackerLightning Model ===")
    model = SiamTrackerLightning(
        student_pretrained=True,
        teacher_weights_path=teacher_weights_path,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        beta_kd=args.beta_kd
    )

    # Step 4: Configure Callbacks
    print("\n=== Step 4: Setting up Callbacks & Trainer ===")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.weights_dir,
        filename="best_siam_tracker",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    
    pruning_callback = SiamPruningCallback(
        pruning_amount=args.pruning_amount
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Determine accelerators (use GPU if available, else CPU)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    
    print(f"Training on device: {accelerator.upper()}")

    # Step 5: Start Training
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, pruning_callback, lr_monitor],
        log_every_n_steps=5,
        val_check_interval=1.0,  # Run validation at the end of every epoch
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches
    )

    print("\n=== Step 5: Launching Trainer.fit() ===")
    trainer.fit(model, datamodule=datamodule)
    
    print("\n=== Training Completed Successfully! ===")
    print(f"Unpruned best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Pruned model saved to: {checkpoint_callback.best_model_path.replace('.ckpt', '_pruned.ckpt')}")

if __name__ == "__main__":
    main()
