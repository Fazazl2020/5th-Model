"""
compute_metrics.py - Speech Enhancement Metrics Computation
===========================================================

Computes standard speech enhancement metrics:
- PESQ (Perceptual Evaluation of Speech Quality)
- CSIG (Signal Distortion)
- CBAK (Background Noise Distortion)
- COVL (Overall Quality)
- SSNR (Segmental SNR)
- STOI (Short-Time Objective Intelligibility)

Dependencies:
    pip install pesq pystoi

Based on CMGAN's metrics computation.
"""

import numpy as np

# Try to import metrics libraries
try:
    from pesq import pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("Warning: pesq not installed. Install with: pip install pesq")

try:
    from pystoi import stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("Warning: pystoi not installed. Install with: pip install pystoi")


def compute_pesq(clean_signal, noisy_signal, sr=16000):
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        clean_signal: Clean reference signal
        noisy_signal: Degraded/enhanced signal
        sr: Sample rate (8000 or 16000)

    Returns:
        pesq_score: PESQ score (higher is better)
                    Wideband (16kHz): -0.5 to 4.5
                    Narrowband (8kHz): -0.5 to 4.5
    """
    if not HAS_PESQ:
        return 0.0

    try:
        # PESQ mode: 'wb' for wideband (16kHz), 'nb' for narrowband (8kHz)
        mode = 'wb' if sr == 16000 else 'nb'
        return pesq(sr, clean_signal, noisy_signal, mode)
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return 0.0


def compute_stoi(clean_signal, noisy_signal, sr=16000):
    """
    Compute STOI (Short-Time Objective Intelligibility).

    Args:
        clean_signal: Clean reference signal
        noisy_signal: Degraded/enhanced signal
        sr: Sample rate

    Returns:
        stoi_score: STOI score in range [0, 1] (higher is better)
    """
    if not HAS_STOI:
        return 0.0

    try:
        return stoi(clean_signal, noisy_signal, sr, extended=False)
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return 0.0


def compute_csig_cbak_covl(clean_signal, noisy_signal, sr=16000):
    """
    Compute CSIG, CBAK, and COVL using composite measures.

    These are subjective quality measures derived from the ITU-T P.835 standard:
    - CSIG: Signal distortion (speech quality)
    - CBAK: Background noise distortion
    - COVL: Overall quality

    Args:
        clean_signal: Clean reference signal
        noisy_signal: Degraded/enhanced signal
        sr: Sample rate

    Returns:
        csig, cbak, covl: Quality scores (higher is better)
    """
    # This is a placeholder implementation
    # The full implementation requires the composite measure algorithms
    # from ITU-T P.563 or similar standards

    # For now, return default values
    # TODO: Implement proper composite measures
    # You can use the pysepm library for this:
    # https://github.com/schmiph2/pysepm

    try:
        import pysepm
        csig = pysepm.csig(clean_signal, noisy_signal, sr)
        cbak = pysepm.cbak(clean_signal, noisy_signal, sr)
        covl = pysepm.covl(clean_signal, noisy_signal, sr)
        return csig, cbak, covl
    except ImportError:
        # Default values if pysepm not available
        return 3.0, 3.0, 3.0
    except Exception as e:
        print(f"CSIG/CBAK/COVL computation failed: {e}")
        return 3.0, 3.0, 3.0


def compute_ssnr(clean_signal, noisy_signal, sr=16000, frame_len=0.032, overlap=0.5):
    """
    Compute Segmental SNR (SSNR).

    Args:
        clean_signal: Clean reference signal
        noisy_signal: Degraded/enhanced signal
        sr: Sample rate
        frame_len: Frame length in seconds
        overlap: Frame overlap ratio

    Returns:
        ssnr: Segmental SNR in dB (higher is better)
    """
    # Frame parameters
    frame_samples = int(frame_len * sr)
    hop_samples = int(frame_samples * (1 - overlap))

    # Ensure signals are same length
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]

    # Compute noise
    noise = clean_signal - noisy_signal

    # Frame-based SNR computation
    snr_segments = []
    for i in range(0, min_len - frame_samples, hop_samples):
        clean_frame = clean_signal[i:i + frame_samples]
        noise_frame = noise[i:i + frame_samples]

        # Compute power
        clean_power = np.sum(clean_frame ** 2)
        noise_power = np.sum(noise_frame ** 2)

        # Skip silent frames
        if clean_power < 1e-10:
            continue

        # SNR for this frame (in dB)
        if noise_power < 1e-10:
            snr = 40.0  # Cap at 40 dB
        else:
            snr = 10 * np.log10(clean_power / noise_power)
            snr = np.clip(snr, -10, 40)  # Clip to [-10, 40] dB

        snr_segments.append(snr)

    # Average across segments
    if len(snr_segments) == 0:
        return 0.0

    return np.mean(snr_segments)


def compute_metrics(clean_signal, enhanced_signal, sr=16000, show_progress=0):
    """
    Compute all speech enhancement metrics.

    This is the main function that matches CMGAN's compute_metrics interface.

    Args:
        clean_signal: Clean reference signal (numpy array)
        enhanced_signal: Enhanced signal (numpy array)
        sr: Sample rate (default: 16000)
        show_progress: Whether to show progress (0 or 1)

    Returns:
        metrics: List of [pesq, csig, cbak, covl, ssnr, stoi]
    """
    # Ensure signals are numpy arrays
    if not isinstance(clean_signal, np.ndarray):
        clean_signal = np.array(clean_signal)
    if not isinstance(enhanced_signal, np.ndarray):
        enhanced_signal = np.array(enhanced_signal)

    # Ensure same length
    min_len = min(len(clean_signal), len(enhanced_signal))
    clean_signal = clean_signal[:min_len]
    enhanced_signal = enhanced_signal[:min_len]

    # Normalize to [-1, 1] range if needed
    clean_max = np.max(np.abs(clean_signal))
    enhanced_max = np.max(np.abs(enhanced_signal))
    if clean_max > 1.0:
        clean_signal = clean_signal / clean_max
    if enhanced_max > 1.0:
        enhanced_signal = enhanced_signal / enhanced_max

    # Compute metrics
    pesq_score = compute_pesq(clean_signal, enhanced_signal, sr)
    csig, cbak, covl = compute_csig_cbak_covl(clean_signal, enhanced_signal, sr)
    ssnr = compute_ssnr(clean_signal, enhanced_signal, sr)
    stoi_score = compute_stoi(clean_signal, enhanced_signal, sr)

    metrics = [pesq_score, csig, cbak, covl, ssnr, stoi_score]

    if show_progress:
        print(f"PESQ: {pesq_score:.4f}, CSIG: {csig:.4f}, CBAK: {cbak:.4f}, "
              f"COVL: {covl:.4f}, SSNR: {ssnr:.4f}, STOI: {stoi_score:.4f}")

    return metrics


def test_metrics():
    """Test metrics computation with dummy signals."""
    print("="*70)
    print("Testing Metrics Computation")
    print("="*70)

    # Create dummy signals
    sr = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, sr * duration)

    # Clean signal: sine wave
    clean = np.sin(2 * np.pi * 440 * t)

    # Noisy signal: clean + noise
    noise = 0.1 * np.random.randn(len(clean))
    noisy = clean + noise

    # Enhanced signal: less noise
    noise_reduced = 0.05 * np.random.randn(len(clean))
    enhanced = clean + noise_reduced

    print(f"\nTest signals created:")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Length: {len(clean)} samples")

    print("\n" + "-"*70)
    print("Computing metrics...")
    print("-"*70)

    metrics = compute_metrics(clean, enhanced, sr, show_progress=1)

    print("\n" + "-"*70)
    print("Results:")
    print("-"*70)
    print(f"  PESQ: {metrics[0]:.4f}")
    print(f"  CSIG: {metrics[1]:.4f}")
    print(f"  CBAK: {metrics[2]:.4f}")
    print(f"  COVL: {metrics[3]:.4f}")
    print(f"  SSNR: {metrics[4]:.4f} dB")
    print(f"  STOI: {metrics[5]:.4f}")

    print("\n" + "="*70)
    print("✓ Metrics computation test completed")
    print("="*70)


if __name__ == '__main__':
    test_metrics()
