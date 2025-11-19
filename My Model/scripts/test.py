"""
test.py - CMGAN-Style Testing Script
======================================

Runs CMGAN-style file-by-file testing using the configuration from configs.py.
Now compatible with CMGAN's testing approach while maintaining checkpoint compatibility!

Usage:
    python test.py

Configuration:
    Edit configs.py to change test settings:
    - test_conf['model_file']: Model checkpoint to use
    - test_conf['save_dir']: Where to save enhanced audio
    - test_conf['compute_metrics']: Whether to compute PESQ, STOI, etc.
    - TEST_CLEAN_DIR: Clean test files
    - TEST_NOISY_DIR: Noisy test files

Important:
    This now uses CMGAN-style file-by-file processing instead of DataLoader!
    STFT parameters (n_fft=320, hop=160) are maintained for checkpoint compatibility.
"""

import sys
import os

# Ensure proper imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils.models import Model
from configs import test_conf, TEST_CLEAN_DIR, TEST_NOISY_DIR


def main():
    """
    Main testing function - CMGAN-style.

    Processes test files one-by-one (like CMGAN) and computes metrics inline.
    """

    print("\n" + "="*70)
    print("SPEECH ENHANCEMENT MODEL - CMGAN-STYLE TESTING")
    print("="*70)
    print("Project: T71a1")
    print("Mode: Testing (CMGAN-style file-by-file processing)")
    print("="*70 + "\n")

    print("Test Configuration:")
    print(f"  Model: {test_conf['model_file']}")
    print(f"  Clean dir: {TEST_CLEAN_DIR}")
    print(f"  Noisy dir: {TEST_NOISY_DIR}")
    print(f"  Output dir: {test_conf['save_dir']}")
    print(f"  Compute metrics: {test_conf['compute_metrics']}")
    print(f"  Save tracks: {test_conf['save_tracks']}")
    print(f"  STFT: n_fft={test_conf['n_fft']}, hop={test_conf['hop_length']}")
    print()

    try:
        # Create model and run testing
        model = Model()
        model.test()

        # Testing completed
        print("\n" + "="*70)
        print("✓ TESTING COMPLETED SUCCESSFULLY")
        print("="*70)
        if test_conf['save_tracks']:
            print(f"Enhanced tracks saved in: {test_conf['save_dir']}")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠ TESTING INTERRUPTED BY USER")
        print("="*70 + "\n")
        sys.exit(0)

    except Exception as e:
        print("\n\n" + "="*70)
        print("✗ ERROR OCCURRED DURING TESTING")
        print("="*70)
        print(f"\nError Type: {type(e).__name__}")
        print(f"Error Message: {e}\n")

        print("="*70)
        print("FULL TRACEBACK:")
        print("="*70)
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")

        sys.exit(1)


if __name__ == '__main__':
    main()
