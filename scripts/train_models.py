"""Brain Tumor Classification - Model Training"""

import os
import sys
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0005

TRAIN_DIR = 'data/Training'
TEST_DIR = 'data/Testing'
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True, seed=42
    )
    
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False, seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    
    print(f"\nTraining: {train_generator.samples}")
    print(f"Validation: {val_generator.samples}")
    print(f"Testing: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator

def create_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    
    return model

def plot_training_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/{model_name}_training_history.png")
    plt.close()

def train_model(model, model_name, train_gen, val_gen):
    print(f"\n{'='*60}\nTRAINING: {model_name}\n{'='*60}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'models/{model_name}_best.h5', monitor='val_accuracy', 
                       save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]
    
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS,
        callbacks=callbacks, verbose=1
    )
    
    plot_training_history(history, model_name)
    
    pd.DataFrame(history.history).to_csv(f'results/{model_name}_history.csv', index=False)
    
    return model, history

def evaluate_model(model, model_name, test_gen):
    print(f"\n{'='*60}\nEVALUATING: {model_name}\n{'='*60}")
    
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    with open(f'results/{model_name}_results.json', 'w') as f:
        json.dump({
            'model_name': model_name,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
    
    return test_acc, test_loss, y_true, y_pred

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*60)
    print("BRAIN TUMOR CLASSIFICATION - TRAINING")
    print("="*60)
    
    train_gen, val_gen, test_gen = create_data_generators()
    
    model = create_simple_cnn()
    model, history = train_model(model, 'SimpleCNN', train_gen, val_gen)
    acc, loss, y_true, y_pred = evaluate_model(model, 'SimpleCNN', test_gen)
    
    pd.DataFrame([{'model': 'SimpleCNN', 'accuracy': acc, 'loss': loss}]).to_csv(
        'results/all_models_summary.csv', index=False
    )
    
    print(f"\n{'='*60}\nCOMPLETED\n{'='*60}")
    print(f"\nFinal Accuracy: {acc:.4f}")
    print(f"Model saved: models/SimpleCNN_best.h5")
    print(f"Results: results/")

if __name__ == "__main__":
    main()
