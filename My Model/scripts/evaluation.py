"""
evaluation.py - CMGAN-Style Evaluation Script
==============================================

ADAPTED FOR YOUR MODEL with the following parameters:
- n_fft: 320 (vs CMGAN's 400)
- hop_length: 160 (vs CMGAN's 100)
- Network expects 161 freq bins (vs CMGAN's 201)
- Uses your trained checkpoint

This follows CMGAN's testing procedure while maintaining compatibility
with your trained model.

Usage:
    python evaluation.py --model_path ./ckpt/models/best_pesq.pt \
                         --test_dir /path/to/test/data \
                         --save_dir ./saved_tracks \
                         --save_tracks True
"""

import numpy as np
from natsort import natsorted
import os
import sys
import argparse

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import torch
import torchaudio
import soundfile as sf

from utils.networks import Net
from utils.models import CheckPoint
from utils.pipeline_modules import power_compress, power_uncompress

# Try to import metrics computation
try:
    from tools.compute_metrics import compute_metrics
    HAS_METRICS = True
except ImportError:
    print("Warning: compute_metrics not found. Metrics will not be computed.")
    print("Install: pip install pesq pystoi")
    HAS_METRICS = False


@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=320, hop=160,
    power=0.3, save_tracks=False, device='cuda'
):
    """
    Enhance a single audio track using CMGAN-style processing.

    MODIFIED FOR YOUR MODEL:
    - Uses n_fft=320, hop=160 (not CMGAN's 400/100)
    - Uses your network architecture (not TSCNet)
    - Maintains same processing pipeline as CMGAN

    Args:
        model: Trained network
        audio_path: Path to noisy audio file
        saved_dir: Directory to save enhanced audio
        cut_len: Maximum length for processing (e.g., 16000 * 16 for 16 seconds)
        n_fft: FFT size (320 for your model)
        hop: Hop length (160 for your model)
        power: Power compression exponent (0.3)
        save_tracks: Whether to save enhanced audio
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        est_audio: Enhanced audio as numpy array
        length: Original audio length
    """
    name = os.path.split(audio_path)[-1]

    # Load audio
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    noisy = noisy.to(device)

    # RMS Normalization (CMGAN-style)
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    # Frame-based padding
    length = noisy.size(-1)
    frame_num = int(np.ceil(length / hop))
    padded_len = frame_num * hop
    padding_len = padded_len - length

    # Audio repetition padding (CMGAN-style, not zero padding)
    if padding_len > 0:
        noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

    # Batch reshaping for long files
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while hop % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    # STFT
    noisy_spec = torch.stft(
        noisy,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(device),
        onesided=True,
        return_complex=True
    )

    # Extract real and imaginary parts
    noisy_real = noisy_spec.real
    noisy_imag = noisy_spec.imag

    # Power compression
    noisy_real_c, noisy_imag_c = power_compress(noisy_real, noisy_imag, power)

    # Stack to [batch, 2, time, freq] format for your model
    # Note: CMGAN uses [batch, 2, freq, time] but your model expects [batch, 2, time, freq]
    noisy_input = torch.stack([noisy_real_c, noisy_imag_c], dim=1)  # [B, 2, T, F]

    # Model inference
    # Your model returns [batch, 2, time, freq]
    est = model(noisy_input, global_step=None)

    # Extract estimated real and imaginary
    est_real = est[:, 0, :, :]  # [B, T, F]
    est_imag = est[:, 1, :, :]  # [B, T, F]

    # Power decompression
    est_real_uc, est_imag_uc = power_uncompress(est_real, est_imag, power)

    # Combine to complex spectrum
    est_spec = torch.complex(est_real_uc, est_imag_uc)

    # iSTFT
    est_audio = torch.istft(
        est_spec,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(device),
        onesided=True,
        length=padded_len
    )

    # Denormalize (CMGAN-style)
    est_audio = est_audio / c

    # Flatten and truncate to original length
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length

    # Save if requested
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir,
               n_fft=320, hop=160, power=0.3, cut_len=16000*16, device='cuda'):
    """
    Evaluate model on test dataset (CMGAN-style).

    Args:
        model_path: Path to model checkpoint
        noisy_dir: Directory containing noisy audio files
        clean_dir: Directory containing clean audio files
        save_tracks: Whether to save enhanced audio
        saved_dir: Directory to save enhanced audio
        n_fft: FFT size (320 for your model)
        hop: Hop length (160 for your model)
        power: Power compression exponent (0.3)
        cut_len: Maximum audio length for processing
        device: Device to run on
    """
    print("\n" + "="*70)
    print("CMGAN-STYLE EVALUATION (Adapted for Your Model)")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Noisy dir: {noisy_dir}")
    print(f"Clean dir: {clean_dir}")
    print(f"Output dir: {saved_dir}")
    print(f"STFT params: n_fft={n_fft}, hop={hop}, power={power}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    # Load model
    print("Loading model...")
    model = Net()
    model = model.to(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    ckpt = CheckPoint()
    ckpt.load(model_path, device)
    model.load_state_dict(ckpt.net_state_dict)
    model.eval()

    print(f"✓ Model loaded from: {model_path}")
    print(f"  Epoch: {ckpt.ckpt_info['cur_epoch'] + 1}")
    print(f"  Best loss: {ckpt.ckpt_info.get('best_loss', 'N/A')}")

    # Create output directory
    if save_tracks:
        os.makedirs(saved_dir, exist_ok=True)
        print(f"✓ Output directory: {saved_dir}\n")

    # Get audio list
    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    audio_list = [f for f in audio_list if f.endswith('.wav')]
    num = len(audio_list)

    print(f"Found {num} audio files to process\n")
    print("="*70)
    print("Processing files...")
    print("="*70)

    # Initialize metrics
    if HAS_METRICS:
        metrics_total = np.zeros(6)  # PESQ, CSIG, CBAK, COVL, SSNR, STOI
        metrics_count = 0

    # Process each file
    for idx, audio in enumerate(audio_list):
        noisy_path = os.path.join(noisy_dir, audio)
        clean_path = os.path.join(clean_dir, audio)

        # Check if clean file exists
        if not os.path.exists(clean_path):
            print(f"⚠ Warning: Clean file not found for {audio}, skipping metrics")
            # Still enhance the file
            est_audio, length = enhance_one_track(
                model, noisy_path, saved_dir, cut_len, n_fft, hop, power, save_tracks, device
            )
            continue

        # Enhance
        est_audio, length = enhance_one_track(
            model, noisy_path, saved_dir, cut_len, n_fft, hop, power, save_tracks, device
        )

        # Compute metrics if available
        if HAS_METRICS:
            clean_audio, sr = sf.read(clean_path)
            assert sr == 16000

            # Ensure same length
            min_len = min(len(clean_audio), len(est_audio))
            clean_audio = clean_audio[:min_len]
            est_audio = est_audio[:min_len]

            try:
                metrics = compute_metrics(clean_audio, est_audio, sr, 0)
                metrics = np.array(metrics)
                metrics_total += metrics
                metrics_count += 1
            except Exception as e:
                print(f"⚠ Warning: Metrics computation failed for {audio}: {e}")

        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == num:
            print(f"  Progress: {idx + 1}/{num} files ({100*(idx+1)/num:.1f}%)")

    # Report results
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    print(f"Processed: {num} files")

    if HAS_METRICS and metrics_count > 0:
        metrics_avg = metrics_total / metrics_count
        print(f"\nAverage Metrics (over {metrics_count} files):")
        print(f"  PESQ: {metrics_avg[0]:.4f}")
        print(f"  CSIG: {metrics_avg[1]:.4f}")
        print(f"  CBAK: {metrics_avg[2]:.4f}")
        print(f"  COVL: {metrics_avg[3]:.4f}")
        print(f"  SSNR: {metrics_avg[4]:.4f} dB")
        print(f"  STOI: {metrics_avg[5]:.4f}")
    else:
        print("\nMetrics not computed (install pesq and pystoi)")

    if save_tracks:
        print(f"\nEnhanced tracks saved to: {saved_dir}")

    print("="*70 + "\n")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='CMGAN-Style Evaluation (Adapted for Your Model)'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='./ckpt/models/best_pesq.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/gdata/fewahab/data/Voicebank+demand/My_train_valid_test/test',
        help='Base test directory (should contain noisy/ and clean/ subdirectories)'
    )
    parser.add_argument(
        '--noisy_dir',
        type=str,
        default=None,
        help='Noisy test directory (overrides test_dir/noisy)'
    )
    parser.add_argument(
        '--clean_dir',
        type=str,
        default=None,
        help='Clean test directory (overrides test_dir/clean)'
    )
    parser.add_argument(
        '--save_tracks',
        type=bool,
        default=True,
        help='Save enhanced tracks'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./saved_tracks_evaluation',
        help='Directory to save enhanced tracks'
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=320,
        help='FFT size (320 for your model, DO NOT CHANGE)'
    )
    parser.add_argument(
        '--hop',
        type=int,
        default=160,
        help='Hop length (160 for your model, DO NOT CHANGE)'
    )
    parser.add_argument(
        '--power',
        type=float,
        default=0.3,
        help='Power compression exponent'
    )
    parser.add_argument(
        '--cut_len',
        type=int,
        default=16000 * 16,
        help='Maximum audio length for processing (in samples)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda or cpu)'
    )

    args = parser.parse_args()

    # Determine noisy and clean directories
    if args.noisy_dir is None:
        args.noisy_dir = os.path.join(args.test_dir, 'noisy')
    if args.clean_dir is None:
        args.clean_dir = os.path.join(args.test_dir, 'clean')

    # Validate directories
    if not os.path.isdir(args.noisy_dir):
        raise FileNotFoundError(f"Noisy directory not found: {args.noisy_dir}")
    if not os.path.isdir(args.clean_dir):
        print(f"Warning: Clean directory not found: {args.clean_dir}")
        print("Will skip metrics computation")

    # Run evaluation
    evaluation(
        model_path=args.model_path,
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        save_tracks=args.save_tracks,
        saved_dir=args.save_dir,
        n_fft=args.n_fft,
        hop=args.hop,
        power=args.power,
        cut_len=args.cut_len,
        device=args.device
    )


if __name__ == "__main__":
    main()
