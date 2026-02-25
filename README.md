# Brain Tumor Classification

CNN-based classification of brain tumor types from MRI scans.

## Dataset

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

**Dataset Structure**:

The dataset contains MRI images organized into 4 classes:
- `glioma` - Glioma brain tumors
- `meningioma` - Meningioma brain tumors  
- `notumor` - Healthy brain scans (no tumor)
- `pituitary` - Pituitary brain tumors

Each class has images split into `Training/` and `Testing/` folders.

![Dataset Distribution](results/class_distribution.png)

![Sample Images](results/sample_images.png)

## Model Architecture

SimpleCNN with 10.7M parameters:
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch Normalization + Dropout
- 2 Dense layers (256, 128 neurons)
- Softmax output (4 classes)

## Results

| Configuration | Accuracy | Loss   | Change  |
|---------------|----------|--------|---------|
| Baseline      | 42.39%   | 3.2599 | -       |
| Fine-tuned    | 46.70%   | 2.8472 | +4.31%  |

**Fine-tuned hyperparameters**: LR=0.0005, Batch=16, Epochs=40

### Per-Class Performance (Fine-tuned)

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Glioma     | 1.00      | 0.14   | 0.24     |
| Meningioma | 1.00      | 0.21   | 0.35     |
| No Tumor   | 0.37      | 0.92   | 0.53     |
| Pituitary  | 0.54      | 0.66   | 0.59     |

![Training History](results/SimpleCNN_training_history.png)

![Confusion Matrix](results/confusion_matrix_finetuned_normalized.png)

## Installation

### Step 1: Install UV

Install [uv](https://github.com/astral-sh/uv) if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Setup Project

```bash
# Create virtual environment and install dependencies
uv sync
```

### Step 3: Download Dataset

Use the automated download script:

```bash
# Make script executable
chmod +x download_dataset.sh

# Run the script (it will guide you through the setup)
./download_dataset.sh
```

**What the script does:**
1. Checks UV installation
2. Sets up virtual environment
3. Installs Kaggle CLI
4. Asks for Kaggle credentials (get from: https://www.kaggle.com/settings → API → Create New Token)
5. Downloads dataset to `./data/`
6. Shows statistics

**Manual download (alternative):**
```bash
uv add kaggle
mkdir -p ~/.kaggle
# Create ~/.kaggle/kaggle.json with: {"username":"your_username","key":"your_api_key"}
chmod 600 ~/.kaggle/kaggle.json
uv run kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ./data --unzip
```

### Legacy Installation (pip)

```bash
pip install -r requirements.txt
```

Requirements: TensorFlow, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Pillow

## Usage

### Data Exploration
```bash
# With uv
uv run python scripts/data_exploration.py

# Or with activated venv
python scripts/data_exploration.py
```

### Training
```bash
# With uv
uv run python scripts/train_models.py

# Or with activated venv
python scripts/train_models.py
```
Training time: ~2 hours. Saves model to `models/SimpleCNN_best.h5`.

### Prediction
```bash
# With uv (example with actual filename)
uv run python scripts/predict_single_image.py data/Testing/glioma/Te-gl_1.jpg

# Or with activated venv
python scripts/predict_single_image.py data/Testing/glioma/Te-gl_1.jpg

# Available folders: glioma, meningioma, notumor, pituitary
```

## Limitations

- Low accuracy (46.70%) insufficient for clinical use
- Poor glioma detection (14% recall)
- Simple architecture inadequate for medical imaging

## Future Work

- Transfer learning (VGG16, ResNet50)
- Class balancing with weighted loss
- Advanced data augmentation
- Ensemble methods

## Additional Resources

- **Dataset Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Repository**: [GitHub](https://github.com/supakornn/brain-tumor-classification)

## License

MIT License - Educational and research purposes.
