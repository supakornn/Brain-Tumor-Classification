---
title: Brain Tumor Classification
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: mit
language:
- en
tags:
- medical-imaging
- brain-tumor
- cnn
- tensorflow
- keras
- image-classification
- computer-vision
- healthcare
- mri-scans
datasets:
- brain-tumor-mri-dataset
metrics:
- accuracy
- precision
- recall
- f1
library_name: tensorflow
pipeline_tag: image-classification
---

# Brain Tumor Classification

CNN-based classification of brain tumor types from MRI scans.

## Dataset

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

4 classes: `glioma`, `meningioma`, `notumor`, `pituitary` — each split into `Training/` and `Testing/`.

![Dataset Distribution](images/class_distribution.png)

![Sample Images](images/sample_images.png)

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

![Training History](images/SimpleCNN_training_history.png)

![Confusion Matrix](images/confusion_matrix_finetuned_normalized.png)

## Usage

```python
from tensorflow import keras

model = keras.models.load_model("model/SimpleCNN_best.h5")
```

## GitHub

[![GitHub](https://img.shields.io/badge/GitHub-supakornn%2Fbrain--tumor--classification-black?logo=github)](https://github.com/supakornn/brain-tumor-classification)

## Limitations

- Low accuracy (46.70%) insufficient for clinical use
- Poor glioma detection (14% recall)
- Simple architecture inadequate for medical imaging

## License

MIT License — Educational and research purposes.
