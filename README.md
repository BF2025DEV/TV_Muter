# TV Muter - Automatic Commercial Detection for Football Games

Automatically detect and mute commercials during football games using machine learning on a Raspberry Pi!

## 🎯 Project Overview

This system uses audio analysis and a lightweight neural network to detect commercial breaks during live TV (specifically optimized for football games). When a commercial is detected, it automatically mutes your TV via HDMI-CEC.

### How It Works

1. **Training Phase** (Mac): Record games, label commercials, train ML model
2. **Deployment Phase** (Raspberry Pi 5): Real-time detection and automatic muting

---

## 📋 Requirements

### Hardware

**For Training (Now):**
- MacBook (for training the model)
- Microphone or audio input device
- Access to football games on TV

**For Deployment (When Pi arrives):**
- Raspberry Pi 5
- HDMI Audio Extractor (~$20)
- USB Audio Adapter (~$10)
- 3.5mm aux cable
- HDMI cables
- TV with HDMI-CEC support

### Software

- Python 3.8 or higher
- macOS (for development)
- Raspberry Pi OS (for deployment)

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd TV_Muter

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

**Note**: On macOS, you may need to install PortAudio for PyAudio:
```bash
brew install portaudio
pip install pyaudio
```

If you have issues with PyAudio, try:
```bash
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```

### Step 2: Test Audio Input

```bash
python scripts/01_test_audio.py
```

This will:
- List available audio input devices
- Let you select your microphone
- Show real-time audio levels
- Verify your setup is working

### Step 3: Collect Training Data

**Important**: Do this while watching 2-4 football games!

```bash
python scripts/02_collect_data.py
```

**During the game:**
- Press **SPACE** when commercials START
- Press **SPACE** again when commercials END (game resumes)
- Press **Q** when done

**Tips:**
- Position your Mac's microphone near the TV speaker
- Make sure the room is relatively quiet
- Audio levels should show activity during the game
- Don't worry about being perfectly precise - within 1-2 seconds is fine
- The script auto-saves every minute

**Recommended**: Collect data from at least 2-3 complete games for best results.

### Step 4: Extract Features

After collecting data from your games:

```bash
python scripts/03_extract_features.py
```

This processes the audio recordings and extracts machine learning features (loudness, pitch, spectral characteristics, etc.)

### Step 5: Train the Model

```bash
python scripts/04_train_model.py
```

This trains a neural network on your labeled data. Training takes 5-15 minutes depending on how much data you collected.

The script will:
- Create train/validation splits
- Train a 1D CNN model
- Save the best model
- Convert to TensorFlow Lite for Pi deployment

### Step 6: Evaluate Performance

```bash
python scripts/05_evaluate_model.py
```

This shows:
- Accuracy, precision, recall metrics
- Confusion matrix visualization
- Prediction timeline plots
- Recommendations for improvement

**Target Accuracy**: Aim for >85% accuracy. If lower, collect more training data.

---

## 📊 Understanding Your Results

### Metrics Explained

- **Accuracy**: Overall correctness (aim for >85%)
- **Precision**: When it says "commercial", how often is it right? (aim for >80%)
- **Recall**: How many commercials does it catch? (aim for >80%)

### What's Good Enough?

- **>90% accuracy**: Excellent! Deploy it!
- **85-90% accuracy**: Good enough for real use
- **75-85% accuracy**: Usable but might need more data
- **<75% accuracy**: Collect more training data

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
audio:
  sample_rate: 16000  # Audio quality
  device_index: null  # Specific audio device (from step 1)

detection:
  confidence_threshold: 0.7  # Higher = fewer false positives
  min_commercial_duration: 15  # Ignore short detections

model:
  sequence_length: 10  # How many seconds of history to use
  epochs: 50  # Training iterations
```

---

## 📁 Project Structure

```
TV_Muter/
├── config.yaml              # Configuration settings
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── data/
│   ├── raw/                # Recorded audio files
│   ├── labels/             # Commercial labels (JSON)
│   └── processed/          # Extracted features
│
├── models/
│   ├── best_model.keras    # Trained Keras model
│   ├── commercial_detector.tflite  # Pi-ready model
│   ├── normalization.npz   # Feature normalization params
│   └── evaluation/         # Performance visualizations
│
├── scripts/
│   ├── 01_test_audio.py        # Test audio capture
│   ├── 02_collect_data.py      # Record and label games
│   ├── 03_extract_features.py  # Process audio
│   ├── 04_train_model.py       # Train ML model
│   └── 05_evaluate_model.py    # Evaluate performance
│
└── lib/
    ├── audio.py            # Audio capture utilities
    ├── features.py         # Feature extraction
    └── utils.py            # Helper functions
```

---

## 🎮 Deployment to Raspberry Pi 5

### When Your Pi Arrives

1. **Install Raspberry Pi OS** on your Pi 5

2. **Copy these files to the Pi:**
   ```bash
   # On your Mac, from the project directory
   scp -r models/ config.yaml lib/ pi@raspberrypi.local:~/TV_Muter/
   ```

3. **Install dependencies on Pi:**
   ```bash
   # On the Pi
   cd ~/TV_Muter
   pip install tensorflow-lite numpy pyaudio pyyaml librosa
   ```

4. **Connect hardware:**
   ```
   Streaming Device → HDMI Extractor → TV
                           ↓ (3.5mm audio)
                      USB Audio Adapter
                           ↓ (USB)
                      Raspberry Pi 5
   ```

5. **Test CEC control:**
   ```bash
   # Install CEC utilities
   sudo apt-get install cec-utils

   # Test mute
   echo 'tx 50:72:03' | cec-client -s -d 1
   ```

6. **Run real-time detection** (script to be created):
   ```bash
   python scripts/06_run_live.py
   ```

---

## 🐛 Troubleshooting

### Audio Issues

**"No audio devices found"**
- Check microphone is connected
- On Mac, check System Preferences → Sound → Input

**"Audio levels are very low"**
- Increase TV volume
- Move microphone closer to TV
- Check microphone isn't muted

### Training Issues

**"Not enough data"**
- Collect data from at least 2 complete games
- Make sure you labeled commercials during collection

**"Low accuracy (<75%)"**
- Collect more training data
- Ensure labels are accurate
- Check audio quality (should be clear, not too much background noise)

### Model Issues

**High false positives (muting during game)**
- Increase `confidence_threshold` in config.yaml
- Collect more diverse training data

**Missing commercials**
- Decrease `confidence_threshold`
- Check `min_commercial_duration` setting

---

## 📝 Tips for Best Results

### During Data Collection

1. **Label accurately**: Be consistent about when you press space
2. **Quiet environment**: Minimize background noise
3. **Good audio levels**: TV should be at normal volume
4. **Watch complete games**: Don't stop/start frequently
5. **Different commercials**: More variety = better model

### For Training

1. **More data is better**: 3-4 games > 1 game
2. **Same network**: Commercials vary by network (CBS, FOX, ESPN)
3. **Recent games**: Commercial patterns may change seasonally

### For Deployment

1. **Test first**: Run during a game and observe before trusting it
2. **Fine-tune threshold**: Adjust based on your tolerance for false positives
3. **Keep training**: Retrain monthly with new data to improve

---

## 🎯 Next Steps After This Weekend

1. ✅ Collect data from 2-3 games this weekend
2. ✅ Train your first model
3. ✅ Evaluate accuracy
4. 📦 Order hardware for Pi deployment
5. 🚀 Deploy to Pi when it arrives
6. 🎨 (Optional) Add music playback during commercials

---

## 🤝 Need Help?

If you run into issues:

1. Check the error messages carefully
2. Verify you completed all previous steps
3. Check file paths are correct
4. Ensure virtual environment is activated
5. Try with a fresh virtual environment

---

## 📈 Future Enhancements

- [ ] Add video analysis for better accuracy
- [ ] Music playback during commercials
- [ ] Web interface for monitoring
- [ ] Support for other sports/TV shows
- [ ] Automatic game detection
- [ ] Mobile app control

---

## 📄 License

This is a personal project. Use and modify as you wish!

---

**Enjoy commercial-free football! 🏈**
