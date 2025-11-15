"""
Utility functions for TV Muter
"""

import yaml
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_labels(labels: List[Dict[str, float]], filepath: str):
    """
    Save commercial labels to JSON file

    Args:
        labels: List of label dicts with 'start', 'end', 'label' keys
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Saved {len(labels)} labels to {filepath}")


def load_labels(filepath: str) -> List[Dict[str, float]]:
    """
    Load commercial labels from JSON file

    Args:
        filepath: Path to label file

    Returns:
        List of label dictionaries
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r') as f:
        labels = json.load(f)

    return labels


def labels_to_binary_sequence(labels: List[Dict[str, float]],
                               duration: float,
                               sample_rate: float = 2.0) -> np.ndarray:
    """
    Convert label list to binary sequence

    Args:
        labels: List of dicts with 'start', 'end', 'label' (0=program, 1=commercial)
        duration: Total duration in seconds
        sample_rate: Samples per second for output sequence

    Returns:
        Binary array where 1=commercial, 0=program
    """
    num_samples = int(duration * sample_rate)
    sequence = np.zeros(num_samples, dtype=np.int8)

    for label in labels:
        start_idx = int(label['start'] * sample_rate)
        end_idx = int(label['end'] * sample_rate)
        label_value = label['label']

        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(num_samples, end_idx)

        sequence[start_idx:end_idx] = label_value

    return sequence


def get_timestamp_str() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1h 23m 45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


class LabelManager:
    """Manage commercial/program labels during data collection"""

    def __init__(self):
        self.labels = []
        self.current_state = 0  # 0 = program, 1 = commercial
        self.last_transition_time = 0.0

    def toggle_state(self, current_time: float):
        """
        Toggle between program and commercial

        Args:
            current_time: Current time in seconds
        """
        # Save the previous segment
        if self.last_transition_time < current_time:
            self.labels.append({
                'start': self.last_transition_time,
                'end': current_time,
                'label': self.current_state
            })

        # Toggle state
        self.current_state = 1 - self.current_state
        self.last_transition_time = current_time

        return "COMMERCIAL" if self.current_state == 1 else "PROGRAM"

    def finalize(self, end_time: float):
        """
        Finalize labels at end of recording

        Args:
            end_time: Final time in seconds
        """
        if self.last_transition_time < end_time:
            self.labels.append({
                'start': self.last_transition_time,
                'end': end_time,
                'label': self.current_state
            })

    def get_labels(self) -> List[Dict[str, float]]:
        """Get all labels"""
        return self.labels

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about labels"""
        if not self.labels:
            return {
                'total_segments': 0,
                'commercial_segments': 0,
                'program_segments': 0,
                'commercial_duration': 0.0,
                'program_duration': 0.0
            }

        commercial_segments = [l for l in self.labels if l['label'] == 1]
        program_segments = [l for l in self.labels if l['label'] == 0]

        commercial_duration = sum(l['end'] - l['start'] for l in commercial_segments)
        program_duration = sum(l['end'] - l['start'] for l in program_segments)

        return {
            'total_segments': len(self.labels),
            'commercial_segments': len(commercial_segments),
            'program_segments': len(program_segments),
            'commercial_duration': commercial_duration,
            'program_duration': program_duration,
            'commercial_percentage': commercial_duration / (commercial_duration + program_duration) * 100
            if (commercial_duration + program_duration) > 0 else 0
        }


def create_sequences(features: np.ndarray,
                     labels: np.ndarray,
                     sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training

    Args:
        features: Feature array (time, features)
        labels: Label array (time,)
        sequence_length: Length of sequences

    Returns:
        Tuple of (X, y) where X is (num_sequences, sequence_length, features)
        and y is (num_sequences,)
    """
    X = []
    y = []

    for i in range(len(features) - sequence_length + 1):
        X.append(features[i:i + sequence_length])
        # Label is the label at the END of the sequence
        y.append(labels[i + sequence_length - 1])

    return np.array(X), np.array(y)
