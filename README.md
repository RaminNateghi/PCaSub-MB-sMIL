# PCaSub-MB-sMIL: Multi-Instance Learning for Predicting PAM50 and PSC Molecular Subtypes of Prostate Cancer [npj Precision Oncology 2025]

A distributed training framework for predicting molecular subtypes of prostate cancer using pathology whole slide images based on 
Multi-branch self-attention-based MIL (MB-sMIL) model and fine-tuning pathology foundational model UNIv2.




## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPUs
- See `requirements.txt` for full dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pcasub-mb-mil.git
cd pcasub-mb-mil

# Install dependencies
pip install -r requirements.txt

## Configuration

Edit `config/config.yaml` to set:
- **Hugging Face Token**: Get from https://huggingface.co/settings/tokens
- **Data Directory**: Path to your slide patches
- **Model Parameters**: Adjust as needed
- **Training Hyperparameters**: Learning rate, epochs, etc.

## Usage

### Training

```bash
# Single GPU
torchrun --nproc_per_node=1 train.py

# Multi-GPU (e.g., 4 GPUs)
torchrun --nproc_per_node=4 train.py

# With custom config
torchrun --nproc_per_node=4 train.py --config path/to/config.yaml
```

### Testing

```bash
# Test with best checkpoint on test set
torchrun --nproc_per_node=4 test.py --checkpoint ./checkpoints/checkpoint.pth

# Test on validation set
torchrun --nproc_per_node=4 test.py --checkpoint ./checkpoints/checkpoint.pth --split val

# Test on training set
torchrun --nproc_per_node=1 test.py --checkpoint ./checkpoints/checkpoint.pth --split train
```

## Model Architecture

- **Foundation Model**: UNI-2 (Vision Transformer)
    - **Feature Dimension**: 1536
- **MIL Aggregator**: Multi-branch self-attention-based MIL (MB-sMIL)
    - **Classification**: multi-class prostate cancer subtyping

## Output

Training produces:
- `checkpoints/checkpoint.pth` - Best model based on validation F1
- `checkpoints/last_checkpoint.pth` - Model from last epoch

## Data Format

The code expects slide patches organized as:
```
/path/to/patches/
├── patient0_slide1_patch1_0.png  # Class 0
├── patient1_slide1_patch2_0.png  # Class 0
├── patient1_slide2_patch1_1.png  # Class 1
└── ...
```

Where filenames follow the pattern: `{patient_id}_{slide_info}-{patch_info}_{class}.png`

## Distributed Training

The project uses PyTorch Distributed Data Parallel (DDP) with NCCL backend. Each GPU processes different patches from each slide, then aggregates features before the MIL forward pass.

## Citation

Prediction of molecular subtypes from histology: AI-driven analysis of prostate cancer morphological patterns and therapeutic implications, npj Precision Oncology, 2025 [Under review]
