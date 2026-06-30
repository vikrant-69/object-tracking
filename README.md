# Siamese Object Tracker v2

A lightweight, hardware-aware Siamese Object Tracker using a hybrid CNN-Transformer architecture, on-device temporal memory, and highly efficient linear cross-attention fusion.

## Architecture Overview

The system is designed to be highly efficient (approx. 5.1M trainable parameters) with static tensor shapes throughout, making it explicitly suitable for edge deployment and compilation (e.g., TFLite, ONNX). 

### Block Diagram

```mermaid
graph TD
    subgraph "Input Pipeline"
        T[Templates (T frames)<br/>127x127x3] --> B1
        S[Search Frame<br/>255x255x3] --> B2
    end

    subgraph "Shared Siamese Backbone (MobileViT-XS Variant)"
        B1[Backbone] --> |"Tokens [T, N, D]"| TM
        B2[Backbone] --> |"Search Tokens [K, V]"| F
        B2 -.-> |"Multi-scale Features"| KD[Distillation Projection]
    end

    subgraph "Temporal Memory Module"
        TM[Token Cache (FIFO)<br/>N=4 frames] --> SSM[Diagonal SSM Cell]
        SSM --> |"Aggregated Target Tokens [Q]"| F
    end

    subgraph "Linear Fusion & Head"
        F[Linear Cross-Attention<br/>O(N) Complexity] --> |"Fused Features"| RPN
        RPN[Lightweight RPN Head] --> CLS[Cls Logits<br/>1x32x32]
        RPN --> REG[Reg Offsets<br/>4x32x32]
    end
    
    subgraph "Teacher Model (Knowledge Distillation)"
        TS[Search Frame] -.-> TNet[Teacher SiamRPN++<br/>ResNet-50]
        TNet -.-> |"Teacher Features"| TLoss[KD Loss]
        KD -.-> TLoss
    end

    CLS --> L1[Focal Loss]
    REG --> L2[GIoU Loss]
```

## Core Modules

### 1. Backbone (`src/models/backbone.py`)
A custom MobileViT-XS variant that blends standard Inverted Residual blocks (MobileNetV2-style) with `LinearAttentionTransformerBlock`s in the deeper stages (stages 4 and 5) utilizing dilated convolutions. It extracts dense token representations for both the template sequence and the search image. Multi-scale feature taps are exposed from stages 2, 3, and 5 for knowledge distillation.

### 2. Temporal Memory (`src/models/temporal_memory.py`)
Replaces static single-frame template caching.
- **Token Cache**: A FIFO buffer holding the last $N$ (default: 4) temporal template frames.
- **Diagonal SSM Cell**: A State-Space Model (S4-diagonal variant) processes the chronological sequence of templates to aggregate temporal context into a single, robust target Query ($Q$) representation.

### 3. Linear Cross-Attention Fusion (`src/models/head.py`)
Replaces traditional depthwise cross-correlation. It performs an $O(N)$ linear cross-attention between the aggregated target tokens ($Q$) and the search region tokens ($K, V$). By using an $ELU(x)+1$ feature map kernel, it avoids the quadratic $O(N^2)$ bottleneck of standard dot-product attention while maintaining static shape computability for hardware compilation.

### 4. RPN Head (`src/models/head.py`)
A lightweight Region Proposal Network using small depthwise separable $3\times3$ convolutions. It predicts dense target presence scores (classification) and bounding box offsets (regression) for every spatial location on the fusion map.

## Training Pipeline (`src/models/lightning_tracker.py`)
The PyTorch Lightning module handles the end-to-end training graph:
- **Focal Loss**: Used for the classification branch to dynamically scale loss based on confidence and handle extreme foreground/background class imbalance.
- **GIoU Loss**: Used for bounding box regression.
- **Multi-scale Feature-level KD**: A heavy ResNet-50 teacher model extracts powerful multi-scale features from the search image. The student's multi-scale taps are projected and aligned with the teacher's features using a weighted MSE loss, masked by a 2D Gaussian spatial attention map centered on the ground-truth target.
- **Quantization-Aware Training (QAT)**: Optional PyTorch `torch.ao.quantization` hooks prepare the model for INT8 export.
- **Pruning**: A callback applies structured pruning automatically at the end of training.

## Data Pipeline (`src/data/`)
The `SiameseTrackingDataset` samples chronologically consistent tracking sequences. It returns $T$ sequential template frames alongside a single search frame sampled from the near future (within `max_search_gap`). Shared spatial data augmentations are applied across the template sequence to enforce spatial consistency.

## Setup & Usage

### Configuration (`src/config.yaml`)
All architectural scales, training hyper-parameters, loss weights, and augmentation settings are controlled via `src/config.yaml`. The parameters are currently scaled to hit a ~5.1M trainable parameter budget.

### Training
Start training using the `train.py` script. Any parameter in `config.yaml` can be directly overridden via CLI arguments.

```bash
# Run full training with the default config
python src/train.py --config src/config.yaml

# Run a quick smoke test for debugging
python src/train.py --config src/config.yaml --max_epochs 2 --limit_train_batches 10 --limit_val_batches 5
```

**Features during training:**
- Automatic downloading of OTB/LaSOT sample sequences and teacher weights to `data/` and `weights/`.
- TensorBoard logging (`lightning_logs/`) including scalar loss metrics and image grid visualizations of the target predictions overlaid on the search frames.
- Automatic best-checkpoint saving and pruning generation.
