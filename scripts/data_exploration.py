"""Brain Tumor Classification - Data Exploration"""

import os
import sys
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

TRAIN_DIR = 'data/Training'
TEST_DIR = 'data/Testing'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def count_images():
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    
    train_counts = {}
    test_counts = {}
    
    for cls in CLASSES:
        train_path = os.path.join(TRAIN_DIR, cls)
        test_path = os.path.join(TEST_DIR, cls)
        train_counts[cls] = len([f for f in os.listdir(train_path) if f.endswith('.jpg')])
        test_counts[cls] = len([f for f in os.listdir(test_path) if f.endswith('.jpg')])
    
    df = pd.DataFrame({
        'Class': CLASSES,
        'Training': [train_counts[cls] for cls in CLASSES],
        'Testing': [test_counts[cls] for cls in CLASSES]
    })
    df['Total'] = df['Training'] + df['Testing']
    
    print("\nImage Count per Class:")
    print(df.to_string(index=False))
    print(f"\nTotal Training: {df['Training'].sum()}")
    print(f"Total Testing: {df['Testing'].sum()}")
    print(f"Total: {df['Total'].sum()}")
    
    return df

def analyze_image_properties():
    print("\n" + "=" * 60)
    print("IMAGE PROPERTIES")
    print("=" * 60)
    
    widths, heights = [], []
    
    for cls in CLASSES:
        class_path = os.path.join(TRAIN_DIR, cls)
        images = [f for f in os.listdir(class_path) if f.endswith('.jpg')][:50]
        
        for img_name in images:
            img = Image.open(os.path.join(class_path, img_name))
            w, h = img.size
            widths.append(w)
            heights.append(h)
    
    print(f"\nWidth - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")
    
    return widths, heights

def visualize_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, split in enumerate(['Training', 'Testing']):
        axes[idx].bar(range(len(CLASSES)), df[split], 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        axes[idx].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Images', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{split} Set', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(range(len(CLASSES)))
        axes[idx].set_xticklabels([cls.replace('_', ' ').title() for cls in CLASSES], 
                                  rotation=45, ha='right')
        
        for i, v in enumerate(df[split]):
            axes[idx].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/class_distribution.png")
    plt.close()

def visualize_sample_images():
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Sample MRI Images', fontsize=16, fontweight='bold', y=0.995)
    
    for i, cls in enumerate(CLASSES):
        class_path = os.path.join(TRAIN_DIR, cls)
        images = [f for f in os.listdir(class_path) if f.endswith('.jpg')][:4]
        
        for j, img_name in enumerate(images):
            img = Image.open(os.path.join(class_path, img_name))
            axes[i][j].imshow(img, cmap='gray')
            axes[i][j].axis('off')
            if j == 0:
                axes[i][j].set_title(cls.replace('_', ' ').title(), 
                                    fontsize=12, fontweight='bold', loc='left')
    
    plt.tight_layout()
    plt.savefig('results/sample_images.png', dpi=300, bbox_inches='tight')
    print("Saved: results/sample_images.png")
    plt.close()

def main():
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("BRAIN TUMOR DATA EXPLORATION")
    print("=" * 60)
    
    df = count_images()
    widths, heights = analyze_image_properties()
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    visualize_class_distribution(df)
    visualize_sample_images()
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
