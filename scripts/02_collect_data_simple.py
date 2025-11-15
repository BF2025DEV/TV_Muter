#!/usr/bin/env python3
"""
Phase 2: Collect Training Data (SIMPLE VERSION)

Simpler version with more reliable keyboard input.
Use this if 02_collect_data.py has keyboard issues.

INSTRUCTIONS:
1. Start the script BEFORE the game starts
2. When prompted, press ENTER to toggle between PROGRAM and COMMERCIAL
3. A new prompt appears after each toggle
4. Type 'quit' and press ENTER when done
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.audio import AudioCapture, AudioRecorder
from lib.utils import load_config, LabelManager, save_labels, get_timestamp_str, format_duration
import time
import threading


class SimpleDataCollector:
    """Simplified data collector with easier keyboard input"""

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
        print("\n💾 Auto-saved labels")

    def handle_toggle(self):
        """Handle toggle between program/commercial"""
        state = self.label_manager.toggle_state(self.elapsed_time)

        # Print notification
        timestamp = format_duration(self.elapsed_time)
        if state == "COMMERCIAL":
            print(f"\n✅ [{timestamp}] Switched to COMMERCIAL 🔇")
        else:
            print(f"\n✅ [{timestamp}] Switched to PROGRAM 📺")

        # Show stats
        stats = self.label_manager.get_stats()
        print(f"   Total segments: {stats['total_segments']}")

    def print_status(self):
        """Print current status"""
        state = "COMMERCIAL 🔇" if self.label_manager.current_state == 1 else "PROGRAM 📺"
        level = self.recorder.get_level()
        bar_length = int(level * 30)
        bar = "█" * bar_length
        time_str = format_duration(self.elapsed_time)

        print(f"\nStatus: {state} | Time: {time_str} | Audio: {bar:<30} {level*100:.0f}%")

    def recording_loop(self):
        """Background loop for recording audio"""
        while self.is_running:
            # Update elapsed time
            self.elapsed_time = time.time() - self.start_time

            # Read audio chunk
            audio_chunk = self.capture.read_chunk()
            self.recorder.add_chunk(audio_chunk)

            # Check for auto-save
            if self.elapsed_time - self.last_autosave > self.autosave_interval:
                self.autosave()
                self.last_autosave = self.elapsed_time

            # Small sleep
            time.sleep(0.01)

    def run(self):
        """Main collection loop"""
        self.setup_audio()
        self.start()

        # Start recording thread
        recording_thread = threading.Thread(target=self.recording_loop, daemon=True)
        recording_thread.start()

        print("\n" + "=" * 60)
        print("INSTRUCTIONS")
        print("=" * 60)
        print("Press ENTER to toggle between PROGRAM ↔ COMMERCIAL")
        print("Type 'quit' and press ENTER to stop")
        print("=" * 60)

        # Initial status
        self.print_status()

        try:
            while self.is_running:
                # Wait for user input
                user_input = input("\nPress ENTER to toggle (or 'quit' to stop): ").strip().lower()

                if user_input == 'quit' or user_input == 'q':
                    self.is_running = False
                    break
                else:
                    # Toggle state
                    self.handle_toggle()
                    self.print_status()

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            self.stop()


def main():
    """Main function"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║   TV MUTER - DATA COLLECTION (SIMPLE VERSION)            ║")
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
    print("   2. Recording starts in PROGRAM mode (watching the game)")
    print("   3. When a commercial starts, press ENTER")
    print("   4. When the game resumes, press ENTER again")
    print("   5. Type 'quit' and press ENTER when done\n")
    print("💡 TIP: This version is simpler - just press ENTER at transitions!")

    input("\n👉 Press ENTER when ready to start recording...")

    # Create collector
    collector = SimpleDataCollector(config)

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
