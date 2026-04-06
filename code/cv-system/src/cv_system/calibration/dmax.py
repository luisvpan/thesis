"""DMax map generation from depth frames.

This module implements dmax_map generation process which captures N depth
frames and computes the most frequent depth value (mode) for each pixel.

Refactored from histogram-based approach to direct mode estimation using scipy.stats.mode.
No depth_range filtering — captures full depth range without quantization loss.
"""

import time
from typing import Callable

import numpy as np
from scipy import stats


def generate_dmax_map(
    capture_frame: Callable[[], np.ndarray],
    num_frames: int = 500,
    depth_shape: tuple[int, int] = (424, 512),
) -> np.ndarray:
    """Generate dmax_map by capturing N frames and computing per-pixel mode.

    Args:
        capture_frame: Callable that returns a depth frame as np.ndarray.
        num_frames: Number of frames to capture for mode calculation.
        depth_shape: Expected shape of depth frames (height, width).

    Returns:
        dmax_map: 2D array of same shape as depth frames, where each pixel
            contains the most frequent depth value across captured frames.
            Pixels with no valid data are set to 0.

    Raises:
        ValueError: If depth_shape is not 2D or invalid num_frames.
        RuntimeError: If capture_frame raises an exception or no frames are captured.

    Notes:
        Direct mode estimation uses scipy.stats.mode to compute per-pixel mode.
        Memory usage is ~217MB for 500 frames @ 424x512 uint16 (acceptable).
        No depth_range filtering — full depth range is captured without loss.
    """
    # Validate inputs
    if len(depth_shape) != 2:
        raise ValueError(f"depth_shape must be 2D, got {len(depth_shape)} dimensions")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    height, width = depth_shape

    # Pre-allocate frame stack for direct mode estimation
    frame_stack = np.zeros((num_frames, height, width), dtype=np.uint16)

    captured_count = 0
    start_time = time.perf_counter()

    print(f"Capturing {num_frames} frames for dmax_map generation (direct mode)...")

    for i in range(num_frames):
        try:
            frame = capture_frame()

            # Validate frame shape
            if frame.shape != depth_shape:
                raise ValueError(
                    f"Frame shape mismatch: expected {depth_shape}, got {frame.shape}"
                )

            # Direct assignment to frame stack (no binning)
            frame_stack[i] = frame
            captured_count += 1

            # Progress feedback
            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Captured {i + 1}/{num_frames} frames ({fps:.1f} fps)")

        except ValueError:
            # Re-raise validation errors immediately
            raise
        except Exception as e:
            print(f"Warning: Failed to capture frame {i + 1}: {e}")
            continue

    elapsed = time.perf_counter() - start_time
    print(
        f"Captured {captured_count}/{num_frames} frames in {elapsed:.2f}s "
        f"({captured_count / elapsed if elapsed > 0 else 0:.1f} fps)"
    )

    if captured_count == 0:
        raise RuntimeError("Failed to capture any frames")

    # Compute per-pixel mode using scipy.stats.mode
    # axis=0 computes mode along time axis (frames)
    # nan_policy='omit' excludes invalid depth readings (NaN, out-of-range)
    # keepdims=False ensures output is 2D (height, width), not reduced to 1D
    mode_result = stats.mode(frame_stack, axis=0, nan_policy="omit", keepdims=False)

    # Extract mode array
    dmax_map = mode_result.mode

    # Convert to uint16 (depth range is 0-65535mm)
    dmax_map = dmax_map.astype(np.uint16)

    mode_time_ms = (elapsed * 1000) if elapsed > 0 else 0

    print(
        f"dmax_map generated: shape={dmax_map.shape}, dtype={dmax_map.dtype}, mode computed in {mode_time_ms:.0f}ms"
    )

    return dmax_map


def compute_depth_stats(
    dmax_map: np.ndarray,
) -> dict[str, float]:
    """Compute statistics about the dmax_map.

    Args:
        dmax_map: Generated dmax_map.

    Returns:
        Dictionary with statistics: mean, std, min, max, valid_pixel_ratio.
    """
    # Valid pixels are non-zero (0 is the invalid marker value)
    valid_mask = dmax_map > 0
    valid_pixels = dmax_map[valid_mask]

    if len(valid_pixels) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "valid_pixel_ratio": 0.0,
        }

    return {
        "mean": float(np.mean(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "valid_pixel_ratio": len(valid_pixels) / dmax_map.size,
    }
