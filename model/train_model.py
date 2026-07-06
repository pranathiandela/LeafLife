"""
LeafLife - Plant Disease Detection Model
========================================
Uses PlantVillage dataset with Transfer Learning (MobileNetV2)
Covers 38 disease classes across 14+ crops

USAGE:
  python model/train_model.py

REQUIREMENTS:
  pip install tensorflow opencv-python numpy matplotlib scikit-learn

DATASET:
  Download PlantVillage from:
  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  Extract to: model/dataset/PlantVillage/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )
    print(f"✅ TensorFlow {tf.__version__} loaded")
except ImportError:
    print("❌ TensorFlow not found. Run: pip install tensorflow")
    sys.exit(1)

# ─── Configuration ─────────────────────────────────────────────
CONFIG = {
    'IMG_SIZE': (224, 224),
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'FINE_TUNE_LR': 1e-5,
    'FINE_TUNE_AT': 100,       # Unfreeze from this layer for fine-tuning
    'VALIDATION_SPLIT': 0.15,
    'TEST_SPLIT': 0.15,
    'DATASET_DIR': 'model/dataset/PlantVillage',
    'MODEL_SAVE': 'model/plant_disease_model.h5',
    'HISTORY_SAVE': 'model/training_history.json',
}

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]
NUM_CLASSES = len(CLASS_NAMES)


def check_dataset():
    if not os.path.exists(CONFIG['DATASET_DIR']):
        print(f"\n⚠️  Dataset not found at: {CONFIG['DATASET_DIR']}")
        print("Please download the PlantVillage dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print(f"  Extract to: {CONFIG['DATASET_DIR']}")
        sys.exit(1)

    classes = os.listdir(CONFIG['DATASET_DIR'])
    print(f"✅ Dataset found: {len(classes)} classes")
    total = sum(len(os.listdir(os.path.join(CONFIG['DATASET_DIR'], c)))
                for c in classes if os.path.isdir(os.path.join(CONFIG['DATASET_DIR'], c)))
    print(f"✅ Total images: {total:,}")
    return total


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        validation_split=CONFIG['VALIDATION_SPLIT']
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['VALIDATION_SPLIT']
    )

    train_gen = train_datagen.flow_from_directory(
        CONFIG['DATASET_DIR'],
        target_size=CONFIG['IMG_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        CONFIG['DATASET_DIR'],
        target_size=CONFIG['IMG_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    print(f"✅ Training samples: {train_gen.samples:,}")
    print(f"✅ Validation samples: {val_gen.samples:,}")
    print(f"✅ Classes detected: {len(train_gen.class_indices)}")

    # Save class mapping
    class_map = {v: k for k, v in train_gen.class_indices.items()}
    with open('model/class_indices.json', 'w') as f:
        json.dump(class_map, f, indent=2)
    print("✅ Class indices saved to model/class_indices.json")

    return train_gen, val_gen


def build_model(num_classes):
    """MobileNetV2 transfer learning model"""
    base_model = MobileNetV2(
        input_shape=(*CONFIG['IMG_SIZE'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze initially

    inputs = keras.Input(shape=(*CONFIG['IMG_SIZE'], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    print(f"✅ Model built: MobileNetV2 + custom head ({num_classes} classes)")
    print(f"   Parameters: {model.count_params():,}")
    return model, base_model


def train_model():
    print("\n" + "="*55)
    print("  LeafLife Plant Disease Detection - Model Training")
    print("="*55 + "\n")

    check_dataset()
    os.makedirs('model', exist_ok=True)
    os.makedirs('model/logs', exist_ok=True)

    train_gen, val_gen = build_data_generators()
    num_classes = len(train_gen.class_indices)

    model, base_model = build_model(num_classes)

    # ── Phase 1: Train head only ───────────────────────────────
    print("\n[Phase 1] Training classification head...")
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    callbacks_phase1 = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint('model/best_model_phase1.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7),
    ]

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────
    print(f"\n[Phase 2] Fine-tuning from layer {CONFIG['FINE_TUNE_AT']}...")
    base_model.trainable = True
    for layer in base_model.layers[:CONFIG['FINE_TUNE_AT']]:
        layer.trainable = False

    print(f"   Trainable layers: {sum(1 for l in base_model.layers if l.trainable)}")

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG['FINE_TUNE_LR']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(CONFIG['MODEL_SAVE'], monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, min_lr=1e-8),
        TensorBoard(log_dir='model/logs', histogram_freq=1)
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG['EPOCHS'],
        initial_epoch=len(history1.history['loss']),
        callbacks=callbacks_phase2,
        verbose=1
    )

    # ── Save final model ───────────────────────────────────────
    model.save(CONFIG['MODEL_SAVE'])
    print(f"\n✅ Model saved to: {CONFIG['MODEL_SAVE']}")

    # ── Combined history ───────────────────────────────────────
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history.get(key, [])
    with open(CONFIG['HISTORY_SAVE'], 'w') as f:
        json.dump(combined, f, indent=2)

    # ── Plot ───────────────────────────────────────────────────
    plot_training_history(combined)

    # ── Final evaluation ───────────────────────────────────────
    print("\n[Evaluation] Running on validation set...")
    results = model.evaluate(val_gen, verbose=1)
    print(f"\n{'='*40}")
    print(f"  Final Validation Accuracy: {results[1]*100:.2f}%")
    print(f"  Final Top-3 Accuracy:      {results[2]*100:.2f}%")
    print(f"{'='*40}\n")

    return model, combined


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['accuracy'], label='Train Acc', color='#22c55e', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Acc', color='#16a34a', linestyle='--', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['loss'], label='Train Loss', color='#ef4444', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', color='#dc2626', linestyle='--', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LeafLife Model Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
    print("✅ Training plot saved to model/training_history.png")
    plt.close()


def predict_single(image_path, model_path=None):
    """Predict disease for a single image (for testing)"""
    import cv2

    if model_path is None:
        model_path = CONFIG['MODEL_SAVE']

    model = keras.models.load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, CONFIG['IMG_SIZE'])
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    top3 = np.argsort(preds)[::-1][:3]

    # Load class map
    if os.path.exists('model/class_indices.json'):
        with open('model/class_indices.json') as f:
            class_map = json.load(f)
        get_name = lambda idx: class_map.get(str(idx), CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else 'Unknown')
    else:
        get_name = lambda idx: CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else 'Unknown'

    print(f"\nPredictions for: {image_path}")
    print("-" * 45)
    for i, idx in enumerate(top3):
        name = get_name(idx)
        print(f"  #{i+1}: {name}")
        print(f"       Confidence: {preds[idx]*100:.2f}%")
    return get_name(top3[0]), float(preds[top3[0]]) * 100


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        if len(sys.argv) < 3:
            print("Usage: python train_model.py --predict <image_path>")
        else:
            predict_single(sys.argv[2])
    else:
        train_model()
