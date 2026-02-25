"""Brain Tumor Classification - Single Image Prediction"""

import os
import sys
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras

IMG_SIZE = (150, 150)
CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
# Folder names: glioma, meningioma, notumor, pituitary

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def predict_image(model_path, image_path):
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"Loading image: {image_path}")
    img_display, img_array = load_and_preprocess_image(image_path)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted: {CLASSES[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nProbabilities:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:20s}: {predictions[0][i]:.2%}")
    print("=" * 60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.imshow(img_display, cmap='gray')
    ax1.axis('off')
    ax1.set_title(f'{CLASSES[predicted_class]}\n{confidence:.1%} confidence', 
                  fontsize=12, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax2.barh(CLASSES, predictions[0], color=colors)
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, f'{predictions[0][i]:.1%}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = 'results/prediction.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.show()
    
    return CLASSES[predicted_class], confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_single_image.py <image_path> [model_path]")
        print("\nExample:")
        print("  python scripts/predict_single_image.py data/Testing/glioma/Te-gl_1.jpg")
        print("\nAvailable folders: glioma, meningioma, notumor, pituitary")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        model_path = sys.argv[2]
    else:
        if os.path.exists('models/SimpleCNN_best.h5'):
            model_path = 'models/SimpleCNN_best.h5'
        else:
            print("Error: No model found. Train first.")
            sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    print("\nBRAIN TUMOR PREDICTION")
    predict_image(model_path, image_path)

if __name__ == "__main__":
    main()
