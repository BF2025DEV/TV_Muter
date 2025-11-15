"""
Audio feature extraction for commercial detection
"""

import numpy as np
import librosa
from typing import Dict, List


class FeatureExtractor:
    """Extract audio features for ML model"""

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 window_size: float = 1.0,
                 hop_size: float = 0.5):
        """
        Initialize feature extractor

        Args:
            sample_rate: Audio sample rate in Hz
            n_mfcc: Number of MFCC coefficients to extract
            window_size: Window size in seconds
            hop_size: Hop size in seconds (overlap between windows)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.window_size = window_size
        self.hop_size = hop_size

        # Calculate window/hop in samples
        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)

    def extract_from_audio(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all features from audio signal

        Args:
            audio: Audio data (1D numpy array)

        Returns:
            Dictionary of features, each is array of shape (num_windows, feature_dim)
        """
        features = {}

        # Split audio into windows
        num_windows = 1 + (len(audio) - self.window_samples) // self.hop_samples

        if num_windows <= 0:
            # Audio too short, return empty features
            return self._empty_features()

        # Pre-allocate arrays
        rms_values = []
        zcr_values = []
        spectral_centroid_values = []
        spectral_rolloff_values = []
        mfcc_values = []

        for i in range(num_windows):
            start = i * self.hop_samples
            end = start + self.window_samples
            window = audio[start:end]

            # RMS Energy (loudness)
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)

            # Zero Crossing Rate (pitch/timbre indicator)
            zcr = np.mean(librosa.zero_crossings(window, pad=False))
            zcr_values.append(zcr)

            # Spectral features
            # Compute spectrum
            spectrum = np.abs(librosa.stft(window, n_fft=512))

            # Spectral Centroid (brightness)
            centroid = librosa.feature.spectral_centroid(S=spectrum, sr=self.sample_rate)
            spectral_centroid_values.append(np.mean(centroid))

            # Spectral Rolloff (frequency distribution)
            rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=self.sample_rate)
            spectral_rolloff_values.append(np.mean(rolloff))

            # MFCCs (voice characteristics)
            mfcc = librosa.feature.mfcc(y=window, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)  # Average over time within window
            mfcc_values.append(mfcc_mean)

        # Convert to arrays
        features['rms'] = np.array(rms_values).reshape(-1, 1)
        features['zcr'] = np.array(zcr_values).reshape(-1, 1)
        features['spectral_centroid'] = np.array(spectral_centroid_values).reshape(-1, 1)
        features['spectral_rolloff'] = np.array(spectral_rolloff_values).reshape(-1, 1)
        features['mfcc'] = np.array(mfcc_values)  # Shape: (num_windows, n_mfcc)

        return features

    def _empty_features(self) -> Dict[str, np.ndarray]:
        """Return empty feature dict"""
        return {
            'rms': np.array([]).reshape(0, 1),
            'zcr': np.array([]).reshape(0, 1),
            'spectral_centroid': np.array([]).reshape(0, 1),
            'spectral_rolloff': np.array([]).reshape(0, 1),
            'mfcc': np.array([]).reshape(0, self.n_mfcc)
        }

    def combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine all features into single array

        Args:
            features: Dictionary of features from extract_from_audio()

        Returns:
            Combined feature array of shape (num_windows, total_features)
        """
        if features['rms'].shape[0] == 0:
            return np.array([])

        # Concatenate all features
        combined = np.concatenate([
            features['rms'],
            features['zcr'],
            features['spectral_centroid'],
            features['spectral_rolloff'],
            features['mfcc']
        ], axis=1)

        return combined

    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order"""
        names = ['rms', 'zcr', 'spectral_centroid', 'spectral_rolloff']
        names.extend([f'mfcc_{i}' for i in range(self.n_mfcc)])
        return names

    def get_feature_dim(self) -> int:
        """Get total number of features"""
        return 4 + self.n_mfcc  # rms + zcr + centroid + rolloff + mfcc


def extract_delta_features(features: np.ndarray) -> np.ndarray:
    """
    Extract delta (first derivative) features

    Args:
        features: Feature array of shape (time, features)

    Returns:
        Delta features of same shape
    """
    if len(features) < 2:
        return np.zeros_like(features)

    # Simple delta: difference between consecutive frames
    delta = np.diff(features, axis=0, prepend=features[0:1])
    return delta


def normalize_features(features: np.ndarray,
                       mean: np.ndarray = None,
                       std: np.ndarray = None) -> tuple:
    """
    Normalize features to zero mean and unit variance

    Args:
        features: Feature array
        mean: Pre-computed mean (if None, compute from features)
        std: Pre-computed std (if None, compute from features)

    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)

    normalized = (features - mean) / std
    return normalized, mean, std
