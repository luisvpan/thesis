"""DMax map generation from depth frames.

This module implements the dmax_map generation process which captures N depth
frames and computes the most frequent depth value (mode) for each pixel within
a configured depth range.

The dmax_map is used in touch detection to distinguish between a baseline
surface depth and touch interactions.
"""

import time
from typing import Callable

import numpy as np


def generate_dmax_map(
    capture_frame: Callable[[], np.ndarray],
    num_frames: int = 500,
    depth_range: tuple[int, int] = (650, 800),
    depth_shape: tuple[int, int] = (424, 512),
) -> np.ndarray:
    """Generate dmax_map by capturing N frames and computing per-pixel mode.

    Args:
        capture_frame: Callable that returns a depth frame as np.ndarray.
        num_frames: Number of frames to capture for mode calculation.
        depth_range: (min_depth, max_depth) inclusive range for valid pixels.
        depth_shape: Expected shape of depth frames (height, width).

    Returns:
        dmax_map: 2D array of same shape as depth frames, where each pixel
            contains the most frequent depth value within the configured range.

    Raises:
        ValueError: If depth_range is invalid or depth_shape is not 2D.
        RuntimeError: If capture_frame raises an exception during capture.
    """
    # Validate inputs
    if len(depth_shape) != 2:
        raise ValueError(f"depth_shape must be 2D, got {len(depth_shape)} dimensions")
    if depth_range[0] >= depth_range[1]:
        raise ValueError(f"depth_range min must be less than max, got {depth_range}")

    min_depth, max_depth = depth_range
    depth_width = max_depth - min_depth + 1

    # Initialize histogram accumulator
    # Shape: (depth_width, height, width)
    histogram = np.zeros((depth_width, *depth_shape), dtype=np.uint32)

    captured_count = 0
    start_time = time.time()

    print(f"Capturing {num_frames} frames for dmax_map generation...")
    print(f"Depth range: {depth_range[0]}-{depth_range[1]}")

    for i in range(num_frames):
        try:
            frame = capture_frame()

            # Validate frame shape
            if frame.shape != depth_shape:
                raise ValueError(
                    f"Frame shape mismatch: expected {depth_shape}, got {frame.shape}"
                )

            # Mask pixels within depth range
            valid_mask = (frame >= min_depth) & (frame <= max_depth)

            # For each valid pixel, increment the corresponding histogram bin
            # We vectorize this by using frame indexing
            frame_clipped = np.clip(frame, min_depth, max_depth)
            bin_indices = frame_clipped - min_depth

            # Efficient histogram update using advanced indexing
            np.add.at(histogram, (bin_indices, *np.indices(depth_shape)), valid_mask)

            captured_count += 1

            # Progress feedback
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                print(f"  Captured {i + 1}/{num_frames} frames ({fps:.1f} fps)")

        except ValueError:
            # Re-raise validation errors immediately
            raise
        except Exception as e:
            print(f"Warning: Failed to capture frame {i + 1}: {e}")
            continue

    elapsed = time.time() - start_time
    print(
        f"Captured {captured_count}/{num_frames} frames in {elapsed:.2f}s "
        f"({captured_count / elapsed:.1f} fps)"
    )

    if captured_count == 0:
        raise RuntimeError("Failed to capture any frames")

    # Compute mode for each pixel: depth value with maximum histogram count
    dmax_map = np.argmax(histogram, axis=0) + min_depth
    dmax_map = dmax_map.astype(np.uint16)

    # Set pixels with no valid data to 0 (invalid)
    max_counts = np.max(histogram, axis=0)
    dmax_map[max_counts == 0] = 0

    print(f"dmax_map generated: shape={dmax_map.shape}, dtype={dmax_map.dtype}")

    return dmax_map


def compute_depth_stats(
    dmax_map: np.ndarray, depth_range: tuple[int, int] = (650, 800)
) -> dict[str, float]:
    """Compute statistics about the dmax_map.

    Args:
        dmax_map: Generated dmax_map.
        depth_range: (min_depth, max_depth) range used for filtering.

    Returns:
        Dictionary with statistics: mean, std, min, max, valid_pixel_ratio.
    """
    valid_mask = (dmax_map >= depth_range[0]) & (dmax_map <= depth_range[1])
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
