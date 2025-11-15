#!/usr/bin/env python3
"""
Phase 4: Train the Commercial Detection Model

This script trains a neural network to detect commercials
based on the features you extracted.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.utils import load_config, create_sequences, ensure_dir
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


def create_model(input_shape, model_config):
    """
    Create 1D CNN model for sequence classification

    Args:
        input_shape: (sequence_length, num_features)
        model_config: Model configuration dict

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=input_shape),

        # 1D Convolutional layers to detect patterns in time
        keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),

        # Output layer
        keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=model_config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


def main():
    """Main function"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║      TV MUTER - MODEL TRAINING (Phase 4)                 ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Load config
    config = load_config()

    # Load processed features
    processed_dir = config['paths']['processed_data']
    features_file = os.path.join(processed_dir, "features.npy")
    labels_file = os.path.join(processed_dir, "labels.npy")

    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        print(f"\n❌ Processed data not found in {processed_dir}")
        print("   Run 03_extract_features.py first!")
        return

    print("\n📂 Loading processed data...")
    features = np.load(features_file)
    labels = np.load(labels_file)

    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Normalize features
    print("\n🔧 Normalizing features...")
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)  # Avoid division by zero

    features_normalized = (features - feature_mean) / feature_std

    # Save normalization parameters
    norm_file = os.path.join(processed_dir, "normalization.npz")
    np.savez(norm_file, mean=feature_mean, std=feature_std)
    print(f"   Saved normalization parameters to {norm_file}")

    # Create sequences
    print("\n🔧 Creating sequences...")
    model_config = config['model']
    sequence_length = model_config['sequence_length']

    X, y = create_sequences(features_normalized, labels, sequence_length)

    print(f"   Sequence shape: {X.shape}")
    print(f"   Label shape: {y.shape}")

    # Check class balance
    commercial_count = np.sum(y)
    program_count = len(y) - commercial_count
    commercial_pct = commercial_count / len(y) * 100

    print(f"\n📊 Dataset balance:")
    print(f"   Commercial: {commercial_count} ({commercial_pct:.1f}%)")
    print(f"   Program: {program_count} ({100-commercial_pct:.1f}%)")

    # Split into train/validation
    print("\n🔧 Splitting into train/validation sets...")
    train_split = model_config['train_split']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_split,
        random_state=42,
        stratify=y  # Maintain class balance
    )

    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")

    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    print(f"\n⚖️  Class weights: {class_weight_dict}")

    # Create model
    print("\n🏗️  Building model...")
    input_shape = (sequence_length, X.shape[2])
    model = create_model(input_shape, model_config)

    print("\n📋 Model architecture:")
    model.summary()

    # Callbacks
    models_dir = config['paths']['models']
    ensure_dir(models_dir)

    checkpoint_path = os.path.join(models_dir, "best_model.keras")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    # Train model
    print("\n🚀 Training model...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=model_config['epochs'],
        batch_size=model_config['batch_size'],
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    results = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Results:")
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]*100:.2f}%")
    print(f"   Precision: {results[2]*100:.2f}%")
    print(f"   Recall: {results[3]*100:.2f}%")

    # Convert to TensorFlow Lite for deployment on Pi
    print("\n🔧 Converting to TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(models_dir, "commercial_detector.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"   Saved TFLite model to {tflite_path}")

    # Save model info
    info_file = os.path.join(models_dir, "model_info.txt")
    with open(info_file, 'w') as f:
        f.write("Commercial Detection Model\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sequence length: {sequence_length}\n")
        f.write(f"Feature dimension: {X.shape[2]}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"\nValidation Performance:\n")
        f.write(f"  Accuracy: {results[1]*100:.2f}%\n")
        f.write(f"  Precision: {results[2]*100:.2f}%\n")
        f.write(f"  Recall: {results[3]*100:.2f}%\n")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n✅ Model training complete!")
    print("\n📊 Evaluate model performance: python scripts/05_evaluate_model.py")
    print("\n🚀 When Pi 5 arrives, transfer these files:")
    print(f"   - {tflite_path}")
    print(f"   - {norm_file}")
    print(f"   - config.yaml")
    print()


if __name__ == "__main__":
    main()
