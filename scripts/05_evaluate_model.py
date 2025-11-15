#!/usr/bin/env python3
"""
Phase 5: Evaluate Model Performance

This script evaluates the trained model and creates visualizations
to help you understand how well it works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.utils import load_config, create_sequences
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Program', 'Commercial'])
    plt.yticks([0.5, 1.5], ['Program', 'Commercial'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"   Saved confusion matrix to {save_path}")
    plt.close()


def plot_prediction_timeline(y_true, y_pred, save_path, max_samples=2000):
    """Plot predictions vs ground truth over time"""
    # Limit samples for readability
    if len(y_true) > max_samples:
        indices = np.linspace(0, len(y_true)-1, max_samples, dtype=int)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    plt.figure(figsize=(14, 6))

    # Plot ground truth
    plt.subplot(2, 1, 1)
    plt.plot(y_true_plot, 'g-', linewidth=1, label='Ground Truth')
    plt.ylabel('Label')
    plt.title('Ground Truth Labels Over Time')
    plt.yticks([0, 1], ['Program', 'Commercial'])
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot predictions
    plt.subplot(2, 1, 2)
    plt.plot(y_pred_plot, 'b-', linewidth=1, label='Predictions')
    plt.ylabel('Label')
    plt.xlabel('Time (samples)')
    plt.title('Model Predictions Over Time')
    plt.yticks([0, 1], ['Program', 'Commercial'])
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"   Saved timeline plot to {save_path}")
    plt.close()


def plot_training_history(models_dir, save_path):
    """Plot training history if available"""
    # Note: This would require saving history during training
    # For now, we'll skip this
    pass


def main():
    """Main function"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║      TV MUTER - MODEL EVALUATION (Phase 5)               ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Load config
    config = load_config()

    # Load processed data
    processed_dir = config['paths']['processed_data']
    models_dir = config['paths']['models']

    features_file = os.path.join(processed_dir, "features.npy")
    labels_file = os.path.join(processed_dir, "labels.npy")
    norm_file = os.path.join(processed_dir, "normalization.npz")
    model_file = os.path.join(models_dir, "best_model.keras")

    # Check files exist
    missing_files = []
    for f in [features_file, labels_file, norm_file, model_file]:
        if not os.path.exists(f):
            missing_files.append(f)

    if missing_files:
        print("\n❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n   Run previous steps first!")
        return

    # Load data
    print("\n📂 Loading data...")
    features = np.load(features_file)
    labels = np.load(labels_file)
    norm_params = np.load(norm_file)

    print(f"   Features: {features.shape}")
    print(f"   Labels: {labels.shape}")

    # Normalize
    print("\n🔧 Normalizing features...")
    features_normalized = (features - norm_params['mean']) / norm_params['std']

    # Create sequences
    print("\n🔧 Creating sequences...")
    sequence_length = config['model']['sequence_length']
    X, y = create_sequences(features_normalized, labels, sequence_length)

    print(f"   Sequences: {X.shape}")

    # Load model
    print("\n📂 Loading model...")
    model = tf.keras.models.load_model(model_file)

    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred_proba = model.predict(X, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Evaluation metrics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    print("\n📊 Classification Report:")
    print(classification_report(y, y_pred, target_names=['Program', 'Commercial']))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\n📊 Confusion Matrix:")
    print("                Predicted")
    print("              Program  Commercial")
    print(f"True Program    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  Commercial    {cm[1,0]:6d}    {cm[1,1]:6d}")

    # Calculate additional metrics
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n📊 Summary Metrics:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1*100:.2f}%")

    # Visualizations
    print("\n📈 Creating visualizations...")
    eval_dir = os.path.join(models_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    plot_confusion_matrix(y, y_pred, os.path.join(eval_dir, "confusion_matrix.png"))
    plot_prediction_timeline(y, y_pred, os.path.join(eval_dir, "predictions_timeline.png"))

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if accuracy >= 0.90:
        print("\n✅ EXCELLENT! Your model is very accurate (>90%)")
        print("   This should work great in real-world use!")
    elif accuracy >= 0.85:
        print("\n✅ GOOD! Your model is fairly accurate (85-90%)")
        print("   Should work well with occasional mistakes")
    elif accuracy >= 0.75:
        print("\n⚠️  OK. Your model is moderately accurate (75-85%)")
        print("   Consider collecting more training data")
    else:
        print("\n❌ NEEDS IMPROVEMENT. Accuracy is below 75%")
        print("   Recommendations:")
        print("   - Collect more training data (2-3 more games)")
        print("   - Make sure labels are accurate")
        print("   - Check audio quality during recording")

    print("\n💡 Tips:")
    if recall < 0.80:
        print("   - Low recall means commercials are being missed")
        print("   - The system might not mute when it should")
    if precision < 0.80:
        print("   - Low precision means false alarms (muting during the game)")
        print("   - Consider increasing the confidence threshold in config.yaml")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n✅ Model evaluation complete!")
    print("\n📊 Check the visualizations in:")
    print(f"   {eval_dir}/")
    print("\n🚀 When Pi 5 arrives:")
    print("   1. Transfer the models/ and config.yaml to the Pi")
    print("   2. Install dependencies on Pi")
    print("   3. Test the real-time detection system")
    print()


if __name__ == "__main__":
    main()
