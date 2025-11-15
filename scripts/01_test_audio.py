#!/usr/bin/env python3
"""
Phase 1: Test Audio Input

This script helps you:
1. Find available audio input devices
2. Test that audio is being captured
3. Verify audio levels

Run this first to make sure your microphone or audio input is working!
"""

import sys
import os

# Add parent directory to path so we can import lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.audio import AudioCapture
from lib.utils import load_config
import time
import numpy as np


def print_audio_devices():
    """List all available audio input devices"""
    print("\n" + "=" * 60)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("=" * 60)

    capture = AudioCapture()
    devices = capture.list_devices()

    if not devices:
        print("❌ No audio input devices found!")
        return None

    for dev in devices:
        print(f"\n[{dev['index']}] {dev['name']}")
        print(f"    Channels: {dev['channels']}")
        print(f"    Sample Rate: {dev['sample_rate']} Hz")

    capture.close()
    return devices


def test_audio_capture(device_index=None):
    """Test audio capture and show real-time levels"""
    print("\n" + "=" * 60)
    print("TESTING AUDIO CAPTURE")
    print("=" * 60)

    # Load config
    try:
        config = load_config()
        sample_rate = config['audio']['sample_rate']
    except:
        sample_rate = 16000

    print(f"\nSample Rate: {sample_rate} Hz")
    print(f"Device: {device_index if device_index is not None else 'Default'}")
    print("\nMake some noise near your microphone/audio input...")
    print("You should see the audio level bars below.\n")
    print("Press Ctrl+C to stop\n")

    try:
        capture = AudioCapture(sample_rate=sample_rate, device_index=device_index)
        capture.start()

        # Capture for a while and show levels
        max_level = 0.0
        chunk_count = 0

        while True:
            # Read audio chunk
            audio_chunk = capture.read_chunk()

            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            max_level = max(max_level, rms)

            # Create visualization bar
            bar_length = int(rms * 50)  # Scale to 50 chars max
            bar = "█" * bar_length
            percentage = rms * 100

            # Print level (overwrite same line)
            print(f"\rLevel: {bar:<50} {percentage:5.1f}%  (Max: {max_level*100:5.1f}%)", end='', flush=True)

            chunk_count += 1

            # Sleep a bit to not spam too fast
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n✅ Audio capture test completed!")
        print(f"   Captured {chunk_count} chunks")
        print(f"   Maximum level: {max_level*100:.1f}%")

        if max_level < 0.01:
            print("\n⚠️  WARNING: Audio levels are very low!")
            print("   - Check that your microphone/input is not muted")
            print("   - Try making louder sounds")
            print("   - You may need to select a different device")
        else:
            print("\n✅ Audio levels look good!")

    finally:
        capture.close()


def main():
    """Main function"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║         TV MUTER - AUDIO INPUT TEST (Phase 1)            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Step 1: List devices
    devices = print_audio_devices()

    if devices is None:
        print("\n❌ Cannot proceed without audio input devices.")
        print("   Please check your microphone/audio input connection.")
        return

    # Step 2: Ask user to select device
    print("\n" + "=" * 60)
    print("DEVICE SELECTION")
    print("=" * 60)

    device_input = input("\nEnter device number to test (or press Enter for default): ").strip()

    if device_input == "":
        device_index = None
        print("Using default device")
    else:
        try:
            device_index = int(device_input)
            print(f"Using device {device_index}")
        except ValueError:
            print("Invalid input, using default device")
            device_index = None

    # Step 3: Test capture
    test_audio_capture(device_index)

    # Step 4: Instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. If the audio test worked, you're ready for Phase 2!")
    print("   Run: python scripts/02_collect_data.py")
    print("\n2. If you want to use a specific device, edit config.yaml:")
    print(f"   Set audio.device_index to {device_index if device_index is not None else 'null'}")
    print("\n3. Make sure to test with your TV audio playing to verify")
    print("   the microphone can pick it up clearly!")
    print()


if __name__ == "__main__":
    main()
