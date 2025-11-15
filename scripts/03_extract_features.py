#!/usr/bin/env python3
"""
Phase 3: Extract Features from Recorded Data

This script processes the audio you recorded and extracts
ML features that will be used to train the model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.audio import load_audio
from lib.features import FeatureExtractor
from lib.utils import load_config, load_labels, labels_to_binary_sequence, ensure_dir
import numpy as np
import glob
from tqdm import tqdm


def process_session(audio_file, labels_file, config):
    """
    Process one recording session

    Returns:
        Tuple of (features, labels) or None if failed
    """
    print(f"\n📂 Processing: {os.path.basename(audio_file)}")

    # Load audio
    print("   Loading audio...")
    audio, sr = load_audio(audio_file, target_sr=config['audio']['sample_rate'])

    if len(audio) == 0:
        print("   ❌ Empty audio file, skipping")
        return None

    duration = len(audio) / sr
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Load labels
    print("   Loading labels...")
    labels_list = load_labels(labels_file)

    if not labels_list:
        print("   ❌ No labels found, skipping")
        return None

    print(f"   Found {len(labels_list)} labeled segments")

    # Extract features
    print("   Extracting features...")
    feature_config = config['features']

    extractor = FeatureExtractor(
        sample_rate=sr,
        n_mfcc=feature_config['n_mfcc'],
        window_size=feature_config['window_size'],
        hop_size=feature_config['hop_size']
    )

    feature_dict = extractor.extract_from_audio(audio)
    features = extractor.combine_features(feature_dict)

    print(f"   Feature shape: {features.shape}")
    print(f"   Feature dim: {extractor.get_feature_dim()}")

    # Convert labels to binary sequence matching feature timestamps
    print("   Aligning labels with features...")

    # Calculate time step for features
    time_step = feature_config['hop_size']  # seconds per feature frame

    # Create binary label sequence
    label_sequence = labels_to_binary_sequence(
        labels_list,
        duration,
        sample_rate=1.0/time_step
    )

    # Trim to match feature length
    min_len = min(len(features), len(label_sequence))
    features = features[:min_len]
    label_sequence = label_sequence[:min_len]

    print(f"   Final length: {len(features)} frames")

    # Calculate statistics
    commercial_frames = np.sum(label_sequence)
    program_frames = len(label_sequence) - commercial_frames
    commercial_pct = commercial_frames / len(label_sequence) * 100

    print(f"   Commercial frames: {commercial_frames} ({commercial_pct:.1f}%)")
    print(f"   Program frames: {program_frames} ({100-commercial_pct:.1f}%)")

    return features, label_sequence


def main():
    """Main function"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║      TV MUTER - FEATURE EXTRACTION (Phase 3)             ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Load config
    config = load_config()

    # Find all recorded sessions
    raw_dir = config['paths']['raw_data']
    labels_dir = config['paths']['labels']
    processed_dir = config['paths']['processed_data']

    ensure_dir(processed_dir)

    # Find audio files
    audio_files = sorted(glob.glob(os.path.join(raw_dir, "game_*.wav")))

    if not audio_files:
        print(f"\n❌ No audio files found in {raw_dir}")
        print("   Run 02_collect_data.py first to record some games!")
        return

    print(f"\n📊 Found {len(audio_files)} recording(s)")

    # Process each session
    all_features = []
    all_labels = []

    for audio_file in audio_files:
        # Find corresponding labels file
        session_id = os.path.basename(audio_file).replace('.wav', '').replace('game_', '')
        labels_file = os.path.join(labels_dir, f"game_{session_id}.json")

        if not os.path.exists(labels_file):
            print(f"\n⚠️  No labels found for {os.path.basename(audio_file)}, skipping")
            continue

        # Process this session
        result = process_session(audio_file, labels_file, config)

        if result is not None:
            features, labels = result
            all_features.append(features)
            all_labels.append(labels)

    if not all_features:
        print("\n❌ No valid data processed!")
        return

    # Combine all sessions
    print("\n" + "=" * 60)
    print("COMBINING ALL SESSIONS")
    print("=" * 60)

    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    print(f"Total features: {combined_features.shape}")
    print(f"Total labels: {combined_labels.shape}")

    # Statistics
    total_commercial = np.sum(combined_labels)
    total_program = len(combined_labels) - total_commercial
    commercial_pct = total_commercial / len(combined_labels) * 100

    print(f"\nDataset statistics:")
    print(f"  Commercial frames: {total_commercial} ({commercial_pct:.1f}%)")
    print(f"  Program frames: {total_program} ({100-commercial_pct:.1f}%)")

    # Save processed data
    print(f"\n💾 Saving processed data...")

    features_file = os.path.join(processed_dir, "features.npy")
    labels_file = os.path.join(processed_dir, "labels.npy")

    np.save(features_file, combined_features)
    np.save(labels_file, combined_labels)

    print(f"   Saved to {processed_dir}/")

    # Save feature names for reference
    feature_config = config['features']
    extractor = FeatureExtractor(
        sample_rate=config['audio']['sample_rate'],
        n_mfcc=feature_config['n_mfcc']
    )
    feature_names = extractor.get_feature_names()

    names_file = os.path.join(processed_dir, "feature_names.txt")
    with open(names_file, 'w') as f:
        for i, name in enumerate(feature_names):
            f.write(f"{i}: {name}\n")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n✅ Feature extraction complete!")
    print("\n▶️  Ready to train model: python scripts/04_train_model.py")
    print()


if __name__ == "__main__":
    main()
