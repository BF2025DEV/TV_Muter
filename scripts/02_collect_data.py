#!/usr/bin/env python3
"""
Phase 2: Collect Training Data

This script records audio from your TV while you watch and lets you
label when commercials start and end.

INSTRUCTIONS:
1. Start the script BEFORE the game starts
2. Press ENTER whenever a commercial starts or ends
3. The script will track everything automatically
4. Type 'q' and press ENTER when done (end of game)

Your labels will be saved automatically!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.audio import AudioCapture, AudioRecorder
from lib.utils import load_config, LabelManager, save_labels, get_timestamp_str, format_duration
import time
import threading


class DataCollector:
    """Manage data collection during a game"""

    def __init__(self, config):
        self.config = config
        self.capture = None
        self.recorder = None
        self.label_manager = LabelManager()

        self.is_running = False
        self.start_time = None
        self.elapsed_time = 0.0

        # For auto-save
        self.last_autosave = 0.0
        self.autosave_interval = config['data_collection']['auto_save_interval']

        # Generate session ID
        self.session_id = get_timestamp_str()

    def setup_audio(self):
        """Initialize audio capture"""
        audio_config = self.config['audio']

        self.capture = AudioCapture(
            sample_rate=audio_config['sample_rate'],
            channels=audio_config['channels'],
            chunk_size=audio_config['chunk_size'],
            device_index=audio_config.get('device_index')
        )

        self.recorder = AudioRecorder(self.capture)

    def start(self):
        """Start recording"""
        print("\n🎬 Starting recording...")
        self.capture.start()
        self.recorder.start_recording()
        self.is_running = True
        self.start_time = time.time()
        print("✅ Recording started in PROGRAM mode!")

    def stop(self):
        """Stop recording"""
        print("\n\n🛑 Stopping recording...")
        self.is_running = False

        # Finalize labels
        self.label_manager.finalize(self.elapsed_time)

        # Save everything
        self.save_data()

        # Cleanup
        self.recorder.stop_recording()
        self.capture.close()

        print("✅ Recording stopped and saved!")

    def save_data(self):
        """Save audio and labels to disk"""
        # Ensure directories exist
        raw_dir = self.config['paths']['raw_data']
        labels_dir = self.config['paths']['labels']
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Save audio
        audio_filename = os.path.join(raw_dir, f"game_{self.session_id}.wav")
        self.recorder.save(audio_filename)

        # Save labels
        labels_filename = os.path.join(labels_dir, f"game_{self.session_id}.json")
        save_labels(self.label_manager.get_labels(), labels_filename)

        # Print stats
        stats = self.label_manager.get_stats()
        print("\n" + "=" * 60)
        print("RECORDING STATISTICS")
        print("=" * 60)
        print(f"Total duration: {format_duration(self.elapsed_time)}")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Commercial segments: {stats['commercial_segments']} ({format_duration(stats['commercial_duration'])})")
        print(f"Program segments: {stats['program_segments']} ({format_duration(stats['program_duration'])})")
        print(f"Commercial %: {stats['commercial_percentage']:.1f}%")
        print("=" * 60)

    def autosave(self):
        """Auto-save labels periodically"""
        labels_dir = self.config['paths']['labels']
        os.makedirs(labels_dir, exist_ok=True)

        temp_filename = os.path.join(labels_dir, f"game_{self.session_id}_autosave.json")

        # Create temporary labels with current state
        temp_labels = self.label_manager.get_labels().copy()
        if self.label_manager.last_transition_time < self.elapsed_time:
            temp_labels.append({
                'start': self.label_manager.last_transition_time,
                'end': self.elapsed_time,
                'label': self.label_manager.current_state
            })

        save_labels(temp_labels, temp_filename)
        print("💾 Auto-saved")

    def handle_toggle(self):
        """Handle ENTER press (toggle commercial/program)"""
        state = self.label_manager.toggle_state(self.elapsed_time)

        # Print notification on new line (don't interfere with status bar)
        timestamp = format_duration(self.elapsed_time)
        if state == "COMMERCIAL":
            print(f"\n[{timestamp}] 📺 → 🔇 Switched to COMMERCIAL")
        else:
            print(f"\n[{timestamp}] 🔇 → 📺 Switched to PROGRAM")

    def print_status(self):
        """Print status update on new line"""
        # Get current state
        state = "COMMERCIAL" if self.label_manager.current_state == 1 else "PROGRAM"
        state_icon = "🔇" if self.label_manager.current_state == 1 else "📺"

        # Get audio level
        level = self.recorder.get_level()
        bar_length = int(level * 30)
        bar = "█" * bar_length

        # Get stats
        stats = self.label_manager.get_stats()

        # Format time
        time_str = format_duration(self.elapsed_time)

        # Print status on NEW line to avoid conflicts with input
        status = (f"[{time_str}] {state_icon} {state:<12} | "
                  f"Level: {bar:<30} | "
                  f"Segments: {stats['total_segments']}")

        print(f"\n{status}")

    def run(self):
        """Main collection loop"""
        self.setup_audio()
        self.start()

        last_status_print = 0.0
        status_interval = 5.0  # Print status every 5 seconds

        try:
            while self.is_running:
                # Update elapsed time
                self.elapsed_time = time.time() - self.start_time

                # Read audio chunk
                audio_chunk = self.capture.read_chunk()
                self.recorder.add_chunk(audio_chunk)

                # Print status periodically (every 5 seconds)
                if self.elapsed_time - last_status_print >= status_interval:
                    self.print_status()
                    last_status_print = self.elapsed_time

                # Check for auto-save
                if self.elapsed_time - self.last_autosave > self.autosave_interval:
                    self.autosave()
                    self.last_autosave = self.elapsed_time

                # Small sleep to prevent crazy CPU usage
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            self.stop()


def input_listener(collector):
    """Listen for ENTER key presses in separate thread"""
    print("\n" + "="*60)
    print("Ready to collect data!")
    print("Press ENTER to toggle PROGRAM ↔ COMMERCIAL")
    print("Type 'q' and press ENTER to quit")
    print("="*60 + "\n")

    while collector.is_running:
        try:
            # Wait for user input (blocking, but in separate thread so it's fine)
            user_input = input(">>> ").strip().lower()

            if user_input == 'q' or user_input == 'quit':
                collector.is_running = False
            else:
                # Any other input (including just pressing ENTER) toggles
                collector.handle_toggle()

        except EOFError:
            # Handle Ctrl+D
            break
        except Exception as e:
            # If input fails, just continue
            time.sleep(0.1)


def main():
    """Main function"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║      TV MUTER - DATA COLLECTION (Phase 2)                ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Load config
    try:
        config = load_config()
    except FileNotFoundError:
        print("❌ config.yaml not found!")
        print("   Make sure you're running from the project root directory")
        return

    print("\n📋 INSTRUCTIONS:")
    print("   1. Make sure your TV audio is playing")
    print("   2. Start this script BEFORE the game begins")
    print("   3. The recording starts in PROGRAM mode (watching the game)")
    print("   4. Press ENTER when a commercial break STARTS")
    print("   5. Press ENTER again when the commercials END (game resumes)")
    print("   6. Repeat step 4-5 throughout the game")
    print("   7. Type 'q' and press ENTER when you're done\n")

    input("\n👉 Press ENTER when ready to start recording...")

    # Create collector
    collector = DataCollector(config)

    # Start input listener thread
    listener_thread = threading.Thread(target=input_listener, args=(collector,), daemon=True)
    listener_thread.start()

    # Run collection
    collector.run()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n✅ Data collection complete!")
    print("\n📦 Your data has been saved to:")
    print(f"   Audio: {config['paths']['raw_data']}/game_{collector.session_id}.wav")
    print(f"   Labels: {config['paths']['labels']}/game_{collector.session_id}.json")
    print("\n🎯 Recommended: Collect data from 2-3 more games for better accuracy")
    print("\n▶️  When ready, run: python scripts/03_extract_features.py")
    print()


if __name__ == "__main__":
    main()
