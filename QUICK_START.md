# 🚀 Quick Start - Get Ready for This Weekend!

## Step-by-Step Setup (30 minutes)

### 1. Install Dependencies (10 minutes)

```bash
# Make sure you're in the TV_Muter directory
cd /Users/justinmbp/Workspace/TV_Muter

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install PortAudio (required for audio capture on Mac)
brew install portaudio

# Install Python packages
pip install -r requirements.txt
```

**Troubleshooting PyAudio installation:**

If the above fails, try:
```bash
pip install --global-option='build_ext' --global-option='-I/opt/homebrew/include' --global-option='-L/opt/homebrew/lib' pyaudio
```

Or on Intel Macs:
```bash
pip install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio
```

### 2. Test Your Audio (5 minutes)

```bash
python scripts/01_test_audio.py
```

- Select your microphone (or just press Enter for default)
- Make some noise and verify you see audio level bars
- Place your Mac near your TV and verify it picks up TV audio

**Tips:**
- TV volume should be at your normal watching level
- Microphone should be 1-3 feet from TV speaker
- You should see the level bars moving when TV plays

### 3. Ready for Game Day!

**Before the game starts:**

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start data collection
python scripts/02_collect_data.py
```

**During the game:**

1. Press **ENTER** to start recording when ready
2. Watch the game normally
3. When commercials start → Press **SPACE**
4. When game resumes → Press **SPACE** again
5. Repeat steps 3-4 throughout the game
6. When game ends → Press **Q** to save and quit

**Important Notes:**

- ✅ The script starts in "PROGRAM" mode (watching the game)
- ✅ You only press SPACE at transitions (game↔commercial)
- ✅ Don't stress about perfect timing - within 2 seconds is fine
- ✅ Script auto-saves every 60 seconds
- ✅ You'll see live status showing current mode and audio levels

**Example Timeline:**

```
Game starts → (already recording in PROGRAM mode)
First commercial → Press SPACE (switches to COMMERCIAL mode)
Game resumes → Press SPACE (switches back to PROGRAM)
Second commercial → Press SPACE (to COMMERCIAL)
Game resumes → Press SPACE (to PROGRAM)
... continue throughout game ...
Game ends → Press Q (saves everything)
```

---

## After Collecting 2-3 Games

### 4. Extract Features

```bash
python scripts/03_extract_features.py
```

This takes 2-5 minutes depending on how much data you collected.

### 5. Train the Model

```bash
python scripts/04_train_model.py
```

This takes 5-15 minutes. You'll see training progress and final accuracy.

### 6. Evaluate Results

```bash
python scripts/05_evaluate_model.py
```

This shows detailed metrics and creates visualizations in `models/evaluation/`.

**Good accuracy?** You're ready to deploy to Pi when it arrives!

**Low accuracy (<75%)?** Collect data from 1-2 more games and retrain.

---

## Games This Weekend

**When are you watching?**

Plan to run the data collection script for each game. More data = better accuracy!

Ideal: 2-3 complete games (about 6-9 hours of labeled data)
Minimum: 1 complete game (about 3 hours)

---

## Quick Command Reference

```bash
# Activate environment (do this every time)
source venv/bin/activate

# Test audio
python scripts/01_test_audio.py

# Collect data (during games)
python scripts/02_collect_data.py

# Process and train (after collecting)
python scripts/03_extract_features.py
python scripts/04_train_model.py
python scripts/05_evaluate_model.py
```

---

## What Files Will Be Created

After your first game collection:

```
data/raw/game_20231115_140000.wav          # Audio recording
data/labels/game_20231115_140000.json      # Your labels
```

After feature extraction:

```
data/processed/features.npy                # ML features
data/processed/labels.npy                  # Binary labels
data/processed/normalization.npz           # Normalization params
```

After training:

```
models/best_model.keras                    # Full model
models/commercial_detector.tflite          # Pi-ready model
models/model_info.txt                      # Model details
```

After evaluation:

```
models/evaluation/confusion_matrix.png     # Visualization
models/evaluation/predictions_timeline.png # Visualization
```

---

## Need Help?

**Import errors?**
- Make sure virtual environment is activated: `source venv/bin/activate`
- Check all packages installed: `pip list`

**No audio detected?**
- Check System Preferences → Sound → Input
- Make sure mic isn't muted
- Try different device in step 1

**Script crashes?**
- Check Python version: `python3 --version` (need 3.8+)
- Try reinstalling: `pip install -r requirements.txt --force-reinstall`

**Questions during the game?**
- Just press Q to safely stop and save
- You can always run it again for the next game

---

## Timeline Summary

**Today:**
- ✅ Install dependencies (30 min)
- ✅ Test audio (5 min)

**This Weekend (During Games):**
- 🎮 Collect data (3 hours per game)
- 🎮 Aim for 2-3 games total

**After Games:**
- 🤖 Extract features (5 min)
- 🤖 Train model (15 min)
- 📊 Evaluate (5 min)

**When Pi Arrives:**
- 🚀 Deploy to Pi
- 🎯 Test on live games
- 🎉 Enjoy commercial-free football!

---

**You're all set! Good luck with data collection! 🏈**
