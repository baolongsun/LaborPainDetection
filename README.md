# Labor Pain Detection

A deep learning project for automatic labor pain intensity assessment from facial images/videos using ResNet-34 and R(2+1)D-18 models with ordinal regression loss.

## Overview

This project implements two neural network architectures for labor pain detection:
- **ResNet-34**: Processes static facial images to predict pain intensity levels (0-3)
- **R(2+1)D-18**: Processes video sequences to capture temporal dynamics of pain expression

Both models incorporate a **Weighted Ordinal Regression Loss** that accounts for both class imbalance and the ordinal nature of pain intensity labels.

## Features

- **Multi-modal support**: Image-based (ResNet-34) and video-based (R(2+1)D-18) pain assessment
- **Ordinal regression**: Custom loss function that penalizes predictions based on distance from true ordinal label
- **Class imbalance handling**: Automatic weighting based on class frequencies
- **Cross-validation**: 6-fold cross-validation with multiple random seeds for robust evaluation
- **Checkpointing**: Automatic saving of best models and epoch checkpoints
- **Logging**: Comprehensive training logs for monitoring and analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) for faster training

### Dependencies
Install required packages:

```bash
pip install torch torchvision pillow tqdm decord
```

Alternatively, install all dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

For optimal performance, install PyTorch with CUDA support from the [official website](https://pytorch.org/).

## Dataset Preparation

The project expects data in the following format:

### File Structure
```
datas/exp/{seed}_6fold/       # For video training (R(2+1)D-18)
exp_simple/{seed}_6fold/      # For image training (ResNet-34)
```

### Label Files
Each fold should have a `train_{fold_idx}.txt` file with the format:
```
/path/to/video_or_image.ext label
/path/to/another/video_or_image.ext label
...
```

Where `label` is an integer from 0-3 representing pain intensity levels.

### Data Organization
- **Images**: RGB images for ResNet-34 training
- **Videos**: Video clips for R(2+1)D-18 training (16 frames sampled per video)

### Configuration Notes

1. **Path Configuration**:
   - The scripts contain hardcoded paths like `root_dir = r'C:'` (ResNet) and `save_dir = f'c:/resnet_checkpoint_...'`
   - Modify these paths according to your system before running
   - For Linux/macOS, change Windows paths (e.g., `C:/`) to appropriate Unix paths

2. **Dataset Paths**:
   - Update `label_file` paths in both scripts to match your dataset location
   - Ensure `root_dir` is set correctly (empty string if using absolute paths in label files)

3. **Checkpoint Directories**:
   - The save directories are created automatically
   - Change `save_dir` patterns if you want different checkpoint organization

## Usage

### Training ResNet-34 (Image-based)

```bash
cd train
python ResNet_34_train.py
```

Configuration parameters in the script:
- `label_file`: Path to training labels (e.g., `./exp_simple/{seed}_6fold/train_{idx}.txt`)
- `root_dir`: Root directory for images (set to `r'C:'` for absolute paths)
- `num_classes`: 4 (pain levels 0-3)
- `batch_size`: 64
- `num_epochs`: 2 (per fold)
- `learning_rate`: 1e-4
- `save_dir`: Checkpoint directory

### Training R(2+1)D-18 (Video-based)

```bash
cd train
python "R(2+1)D-18_train.py"
```

Configuration parameters:
- `label_file`: Path to video labels (e.g., `./datas/exp/{seed}_6fold/train_{idx}.txt`)
- `root_dir`: Root directory for videos (empty if using absolute paths)
- `num_frames`: 16 (frames per video)
- `batch_size`: 8
- `num_epochs`: 4 (per fold)
- `learning_rate`: 1e-4
- `save_dir`: Checkpoint directory

### Training Process

Both scripts implement:
1. **6-fold cross-validation**: Trains on 5 folds, validates on 1 (repeated for each fold)
2. **Multiple random seeds**: Ensures statistical robustness (10+ seeds)
3. **Class weight calculation**: Automatically balances loss based on class frequencies
4. **Checkpoint saving**: Best model and periodic epoch checkpoints
5. **Logging**: Training progress saved to `{save_dir}/training.log`

## Model Architectures

### ResNet-34 with Custom Head
- Base: Pretrained ResNet-34 from torchvision
- Modification: Final fully-connected layer replaced with 4-class classifier
- Input: 112×112 RGB images
- Preprocessing: Resize to 128×171, center crop, normalization (mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225])

### R(2+1)D-18 for Video Analysis
- Base: Pretrained R(2+1)D-18 from torchvision
- Modification: Final fully-connected layer replaced with 4-class classifier
- Input: 16-frame video clips (112×112 per frame)
- Preprocessing: Same as ResNet-34, applied frame-wise

### Loss Functions
1. **CrossEntropyLoss**: Standard classification loss with class weights
2. **WeightedOrdinalRegressionLoss**: Custom loss that:
   - Computes expected pain score from softmax probabilities
   - Penalizes predictions based on distance from true label (L1/L2 distance)
   - Incorporates class weights for imbalance correction

The total loss is: `Loss = CrossEntropyLoss + WeightedOrdinalRegressionLoss`

## Project Structure

```
LaborPainDetection/
├── train/
│   ├── ResNet_34_train.py      # Image-based training script
│   └── R(2+1)D-18_train.py     # Video-based training script
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .gitignore
```

### Key Components

**ResNet_34_train.py**:
- `ImageDataset`: Custom Dataset for loading images from label files
- `ResNetModel`: ResNet-34 wrapper with customizable classifier
- `WeightedOrdinalRegressionLoss`: Ordinal regression loss implementation
- `train_model()`: Training loop with checkpointing

**R(2+1)D-18_train.py**:
- `VideoDataset`: Custom Dataset for loading videos using decord
- `R2Plus1DModel`: R(2+1)D-18 wrapper with customizable classifier
- `load_video()`: Function to sample frames from videos
- Same loss functions and training loop as ResNet version

## Results & Evaluation

The project uses 6-fold cross-validation with multiple random seeds to ensure robust performance evaluation. Metrics logged during training:

- **Loss**: Combined cross-entropy and ordinal regression loss
- **Accuracy**: Classification accuracy (exact match)

For ordinal regression tasks, additional metrics could be considered:
- Mean Absolute Error (MAE) between predicted and true ordinal scores
- Quadratic Weighted Kappa (QWK) for ordinal agreement

## Future Improvements

1. **Multi-task learning**: Combine pain detection with other physiological signals
2. **Attention mechanisms**: Visual attention to focus on pain-relevant facial regions
3. **Temporal modeling**: LSTMs or Transformers for better sequence modeling in videos
4. **Explainability**: Grad-CAM or attention visualization to interpret model decisions
5. **Real-time inference**: Optimized model deployment for clinical use

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). A closer look at spatiotemporal convolutions for action recognition. CVPR.
- Ordinal regression for pain intensity assessment: Related works in medical image analysis.

## License

This project is for research purposes. Please consult with the authors for licensing information.

## Contact

For questions about the code or dataset, please contact the project maintainers.