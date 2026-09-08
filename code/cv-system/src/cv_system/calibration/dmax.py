"""Surface depth map (dsurface) generation from depth frames.

This module implements dsurface_map generation process which captures N depth
frames and computes the 95th percentile depth value for each pixel.

Uses 95th percentile to capture the FAR edge of surface noise, ensuring that
the actual surface depth is reliably represented. This follows the approach
described in Wilson 2010 "Using a Depth Camera as a Touch Sensor".
"""

import time
from typing import Callable

import cv2
import numpy as np


def generate_dmax_map(
    capture_frame: Callable[[], np.ndarray],
    num_frames: int = 500,
    depth_shape: tuple[int, int] = (424, 512),
    percentile: float = 95,
    surface_offset: int = 3,
    *,
    show_debug: bool = False,
) -> np.ndarray:
    """Generate dmax_map by capturing N frames and computing per-pixel percentile.

    The process:
    1. Capture N frames and compute 95th percentile = dsurface (surface depth)
    2. Apply offset: dmax = dsurface - surface_offset

    This ensures the touch zone (dmin < z < dmax) excludes the actual surface.

    Args:
        capture_frame: Callable that returns a depth frame as np.ndarray.
        num_frames: Number of frames to capture for percentile calculation.
        depth_shape: Expected shape of depth frames (height, width).
        percentile: Percentile to use (default 95). Higher values capture the
            far edge of surface noise.
        surface_offset: Offset in mm to subtract from dsurface to get dmax.
            Higher values = touch zone farther from surface (more conservative).
        show_debug: If True, display a histogram of depth values with percentile markers.

    Returns:
        dmax_map: 2D array where dmax = dsurface - surface_offset.
            Ready to use directly in touch detection.

    Raises:
        ValueError: If depth_shape is not 2D or invalid num_frames.
        RuntimeError: If capture_frame raises an exception or no frames are captured.

    Notes:
        Uses 95th percentile per Wilson 2010 to capture the FAR edge of ToF noise.
        Memory usage is ~217MB for 500 frames @ 424x512 uint16 (acceptable).
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
            # Convert UMat to numpy if needed
            if isinstance(frame, cv2.UMat):
                frame = frame.get()

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

    # Compute per-pixel percentile (captures far edge of surface noise)
    # Replace 0 values with NaN to exclude invalid readings
    print(f"Computing per-pixel {percentile}th percentile...")
    t_start = time.perf_counter()

    frame_stack_float = frame_stack.astype(np.float32)
    frame_stack_float[frame_stack_float == 0] = np.nan

    # Compute percentile along time axis, ignoring NaN values
    with np.errstate(all='ignore'):  # Suppress warnings for all-NaN slices
        dmax_map = np.nanpercentile(frame_stack_float, percentile, axis=0)

    # Replace NaN with 0 for pixels that had no valid readings
    dsurface_map = np.nan_to_num(dmax_map, nan=0.0)

    # Apply surface_offset: dmax = dsurface - offset
    # This ensures the actual surface is excluded from the touch zone
    dmax_map = (dsurface_map - surface_offset).astype(np.int32)
    dmax_map = np.clip(dmax_map, 0, 65535).astype(np.uint16)

    t_end = time.perf_counter()
    compute_time_ms = (t_end - t_start) * 1000

    print(
        f"dmax_map generated: shape={dmax_map.shape}, dtype={dmax_map.dtype}, "
        f"p{percentile:.0f} - {surface_offset}mm offset, computed in {compute_time_ms:.0f}ms"
    )

    # Debug histogram visualization
    if show_debug:
        _show_depth_histogram(
            frame_stack_float, dsurface_map, percentile, surface_offset
        )

    return dmax_map


def generate_dmax_map_wilson(
    capture_frame: Callable[[], np.ndarray],
    num_frames: int = 500,
    depth_shape: tuple[int, int] = (424, 512),
    histogram_threshold: int = 3,
    histogram_range: int = 20,
    surface_offset: int = 2,
    *,
    show_debug: bool = False,
) -> np.ndarray:
    """Generate dmax_map using Wilson 2010 per-pixel histogram algorithm.

    The process (per Wilson 2010):
    1. Capture a snapshot frame as reference
    2. Capture N frames and build per-pixel histograms around the snapshot
    3. For each pixel: scan from NEAR to FAR, find first bin with count >= threshold
    4. That value = dsurface for that pixel
    5. Apply offset: dmax = dsurface - surface_offset

    This adapts to per-pixel noise characteristics, unlike fixed percentile methods.

    Args:
        capture_frame: Callable that returns a depth frame as np.ndarray.
        num_frames: Number of frames to capture for histogram building.
        depth_shape: Expected shape of depth frames (height, width).
        histogram_threshold: Minimum count for a bin to be considered valid.
        histogram_range: Range in mm around snapshot for histogram bins.
            Creates (2 * histogram_range + 1) bins per pixel.
        surface_offset: Offset in mm to subtract from dsurface to get dmax.
        show_debug: If True, display debug visualizations.

    Returns:
        dmax_map: 2D array where dmax = dsurface - surface_offset.
            Ready to use directly in touch detection.

    Raises:
        ValueError: If depth_shape is not 2D or invalid parameters.
        RuntimeError: If capture_frame fails or no frames captured.
    """
    # Validate inputs
    if len(depth_shape) != 2:
        raise ValueError(f"depth_shape must be 2D, got {len(depth_shape)} dimensions")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if histogram_threshold <= 0:
        raise ValueError(f"histogram_threshold must be positive, got {histogram_threshold}")
    if histogram_range <= 0:
        raise ValueError(f"histogram_range must be positive, got {histogram_range}")

    height, width = depth_shape
    num_bins = 2 * histogram_range + 1

    # Step 1: Capture snapshot frame as reference
    print("Capturing snapshot frame as reference...")
    snapshot = capture_frame()
    # Convert UMat to numpy if needed
    if isinstance(snapshot, cv2.UMat):
        snapshot = snapshot.get()
    if snapshot.shape != depth_shape:
        raise ValueError(f"Snapshot shape mismatch: expected {depth_shape}, got {snapshot.shape}")
    snapshot = snapshot.astype(np.int32)

    # Step 2: Initialize per-pixel histograms
    # Shape: (height, width, num_bins) - uses ~35MB for 424x512x41
    print(f"Initializing per-pixel histograms ({num_bins} bins per pixel)...")
    histograms = np.zeros((height, width, num_bins), dtype=np.uint16)

    # Step 3: Capture frames and accumulate histograms
    print(f"Capturing {num_frames} frames for histogram accumulation...")
    start_time = time.perf_counter()
    captured_count = 0

    for i in range(num_frames):
        try:
            frame = capture_frame()
            # Convert UMat to numpy if needed
            if isinstance(frame, cv2.UMat):
                frame = frame.get()
            if frame.shape != depth_shape:
                raise ValueError(f"Frame shape mismatch: expected {depth_shape}, got {frame.shape}")

            # Calculate offset from snapshot (frame - snapshot)
            # Positive offset = frame is farther (higher depth value)
            offset = frame.astype(np.int32) - snapshot

            # Map offset to bin index: offset of -histogram_range -> bin 0, 0 -> bin histogram_range
            bin_indices = np.clip(offset + histogram_range, 0, num_bins - 1).astype(np.int32)

            # Vectorized histogram increment using np.add.at
            y_idx, x_idx = np.ogrid[:height, :width]
            np.add.at(histograms, (y_idx, x_idx, bin_indices), 1)

            captured_count += 1

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Captured {i + 1}/{num_frames} frames ({fps:.1f} fps)")

        except ValueError:
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

    # Step 4: Find dsurface for each pixel using Wilson algorithm
    # Scan from bin 0 (NEAR = snapshot - histogram_range) to bin num_bins-1 (FAR)
    # Find first bin where count >= threshold
    print(f"Computing dsurface using Wilson algorithm (threshold={histogram_threshold})...")
    t_start = time.perf_counter()

    # Create mask where count >= threshold
    valid_mask = histograms >= histogram_threshold

    # Find first valid bin index per pixel (scanning from 0 = near)
    # np.argmax returns 0 when no True values, so we need to handle that case
    first_valid_bin = np.argmax(valid_mask, axis=2)

    # Check if any valid bin was found (argmax returns 0 when no True exists)
    any_valid = np.any(valid_mask, axis=2)

    # Calculate dsurface: snapshot + (first_valid_bin - histogram_range)
    # first_valid_bin = 0 means offset = -histogram_range (near)
    # first_valid_bin = histogram_range means offset = 0 (at snapshot)
    dsurface_map = snapshot + (first_valid_bin - histogram_range)

    # Handle pixels with no valid bins: set to 0 (invalid)
    dsurface_map[~any_valid] = 0

    # Ensure non-negative values
    dsurface_map = np.clip(dsurface_map, 0, 65535).astype(np.uint16)

    # Step 5: Apply surface_offset to get dmax
    dmax_map = (dsurface_map.astype(np.int32) - surface_offset)
    dmax_map = np.clip(dmax_map, 0, 65535).astype(np.uint16)

    t_end = time.perf_counter()
    compute_time_ms = (t_end - t_start) * 1000

    # Statistics
    valid_pixels = dsurface_map > 0
    valid_count = np.sum(valid_pixels)
    valid_ratio = valid_count / dsurface_map.size

    print(
        f"dmax_map generated (Wilson): shape={dmax_map.shape}, dtype={dmax_map.dtype}, "
        f"valid_pixels={valid_ratio:.1%}, computed in {compute_time_ms:.0f}ms"
    )

    # Debug visualizations
    if show_debug:
        _show_dsurface_heatmap(dsurface_map, dmax_map, histogram_range)
        _show_depth_histogram_wilson(
            histograms, dsurface_map, snapshot, histogram_range, histogram_threshold, surface_offset
        )

    return dmax_map


def _show_dsurface_heatmap(
    dsurface_map: np.ndarray,
    dmax_map: np.ndarray,
    histogram_range: int,
) -> None:
    """Display dsurface_map, dmax_map, and invalid pixels as heatmaps.

    Colors: Blue = near (low depth), Red = far (high depth), Black = invalid.
    """
    # Find valid range
    valid_mask = dsurface_map > 0
    invalid_mask = ~valid_mask

    if not np.any(valid_mask):
        print("[Debug] No valid pixels in dsurface_map")
        return

    vmin = np.min(dsurface_map[valid_mask])
    vmax = np.max(dsurface_map[valid_mask])

    if vmax <= vmin:
        print(f"[Debug] dsurface range too small: {vmin}-{vmax}")
        return

    height, width = dsurface_map.shape
    valid_count = np.sum(valid_mask)
    invalid_count = np.sum(invalid_mask)
    valid_ratio = valid_count / dsurface_map.size

    # Create 3-panel visualization: dsurface | dmax | invalid pixels
    combined = np.zeros((height, width * 3 + 40, 3), dtype=np.uint8)

    # Panel 1: dsurface heatmap
    normalized = np.zeros_like(dsurface_map, dtype=np.float32)
    normalized[valid_mask] = (dsurface_map[valid_mask] - vmin) / (vmax - vmin)
    heatmap_dsurface = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_dsurface[invalid_mask] = [0, 0, 0]

    # Panel 2: dmax heatmap
    valid_mask_dmax = dmax_map > 0
    normalized_dmax = np.zeros_like(dmax_map, dtype=np.float32)
    normalized_dmax[valid_mask_dmax] = (dmax_map[valid_mask_dmax] - vmin) / (vmax - vmin)
    heatmap_dmax = cv2.applyColorMap((normalized_dmax * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_dmax[~valid_mask_dmax] = [0, 0, 0]

    # Panel 3: Invalid pixels visualization
    # Green = valid, Magenta = invalid (no valid histogram bin found)
    invalid_vis = np.zeros((height, width, 3), dtype=np.uint8)
    invalid_vis[valid_mask] = [0, 100, 0]       # Dark green = valid
    invalid_vis[invalid_mask] = [255, 0, 255]   # Magenta = invalid

    # Place in combined image
    combined[:, :width] = heatmap_dsurface
    combined[:, width + 20:width * 2 + 20] = heatmap_dmax
    combined[:, width * 2 + 40:] = invalid_vis

    # Add labels
    cv2.putText(combined, "dsurface_map", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, "dmax_map", (width + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combined, f"INVALID ({invalid_count}px = {1-valid_ratio:.1%})",
                (width * 2 + 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    cv2.putText(combined, f"Range: {vmin:.0f}-{vmax:.0f}mm", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, f"Valid: {valid_count}px = {valid_ratio:.1%}",
                (width + 30, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.namedWindow("Wilson Calibration - Heatmaps", cv2.WINDOW_NORMAL)
    cv2.imshow("Wilson Calibration - Heatmaps", combined)
    print(f"[Debug] Showing heatmaps. Valid={valid_ratio:.1%}, Invalid={1-valid_ratio:.1%}. Press any key...")
    cv2.waitKey(0)
    cv2.destroyWindow("Wilson Calibration - Heatmaps")


def _show_depth_histogram_wilson(
    histograms: np.ndarray,
    dsurface_map: np.ndarray,
    snapshot: np.ndarray,
    histogram_range: int,
    histogram_threshold: int,
    surface_offset: int,
) -> None:
    """Display 4 histograms (one per quadrant) showing Wilson algorithm results.

    Shows for each quadrant:
    - Aggregated histogram of all pixels in that quadrant
    - Threshold line
    - dsurface and dmax markers
    """
    height, width, num_bins = histograms.shape
    mid_h, mid_w = height // 2, width // 2

    quadrants = [
        ("Top-Left", slice(0, mid_h), slice(0, mid_w)),
        ("Top-Right", slice(0, mid_h), slice(mid_w, width)),
        ("Bottom-Left", slice(mid_h, height), slice(0, mid_w)),
        ("Bottom-Right", slice(mid_h, height), slice(mid_w, width)),
    ]

    hist_w, hist_h = 400, 200
    padding = 10
    img_width = hist_w * 2 + padding * 3
    img_height = (hist_h + 60) * 2 + padding * 3
    combined_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for idx, (name, row_slice, col_slice) in enumerate(quadrants):
        # Aggregate histogram for this quadrant (sum across pixels)
        quad_hist = histograms[row_slice, col_slice, :]
        aggregated_hist = np.sum(quad_hist, axis=(0, 1))

        # Get dsurface stats for this quadrant
        quad_dsurface = dsurface_map[row_slice, col_slice]
        quad_snapshot = snapshot[row_slice, col_slice]
        valid_ds = quad_dsurface[quad_dsurface > 0]
        valid_snap = quad_snapshot[quad_dsurface > 0]

        dsurface_mean = np.mean(valid_ds) if len(valid_ds) > 0 else 0
        snapshot_mean = np.mean(valid_snap) if len(valid_snap) > 0 else 0
        dmax_mean = dsurface_mean - surface_offset

        # Create histogram image
        hist_img = np.zeros((hist_h + 60, hist_w, 3), dtype=np.uint8)

        # Normalize histogram
        max_count = np.max(aggregated_hist) if np.max(aggregated_hist) > 0 else 1
        hist_norm = aggregated_hist / max_count

        # Draw bars
        bar_w = hist_w // num_bins
        for i, h in enumerate(hist_norm):
            x1, x2 = i * bar_w, (i + 1) * bar_w - 1
            y1 = int(hist_h * (1 - h))
            # Color bars above threshold green, below gray
            color = (0, 200, 0) if aggregated_hist[i] >= histogram_threshold else (80, 80, 80)
            cv2.rectangle(hist_img, (x1, y1), (x2, hist_h), color, -1)

        # Draw threshold line
        threshold_y = int(hist_h * (1 - histogram_threshold / max_count))
        cv2.line(hist_img, (0, threshold_y), (hist_w, threshold_y), (0, 0, 255), 1)

        # Mark snapshot position (bin = histogram_range)
        snapshot_x = histogram_range * bar_w + bar_w // 2
        cv2.line(hist_img, (snapshot_x, 0), (snapshot_x, hist_h), (255, 255, 0), 2)

        # Calculate dsurface bin position
        # dsurface = snapshot + (first_valid_bin - histogram_range)
        # So first_valid_bin = dsurface - snapshot + histogram_range
        dsurface_offset = dsurface_mean - snapshot_mean
        dsurface_bin = int(dsurface_offset + histogram_range)
        dsurface_bin = np.clip(dsurface_bin, 0, num_bins - 1)
        dsurface_x = dsurface_bin * bar_w + bar_w // 2
        cv2.line(hist_img, (dsurface_x, 0), (dsurface_x, hist_h), (0, 255, 255), 2)

        # Title and info
        cv2.putText(hist_img, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(hist_img, f"snapshot: {snapshot_mean:.0f}",
                    (5, hist_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        cv2.putText(hist_img, f"dsurface: {dsurface_mean:.0f}",
                    (5, hist_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(hist_img, f"dmax: {dmax_mean:.0f}",
                    (5, hist_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(hist_img, f"thresh: {histogram_threshold}",
                    (150, hist_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Place in combined image
        row, col = idx // 2, idx % 2
        y_off = padding + row * (hist_h + 60 + padding)
        x_off = padding + col * (hist_w + padding)
        combined_img[y_off:y_off + hist_h + 60, x_off:x_off + hist_w] = hist_img

    # Add legend
    legend_y = img_height - 25
    cv2.putText(combined_img, "Bins: -range...0...+range | ",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(combined_img, "snapshot", (220, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    cv2.putText(combined_img, "dsurface", (290, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    cv2.putText(combined_img, "threshold", (360, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.namedWindow("Wilson Calibration - Quadrant Histograms", cv2.WINDOW_NORMAL)
    cv2.imshow("Wilson Calibration - Quadrant Histograms", combined_img)
    print("[Debug] Showing Wilson histograms. Press any key...")
    cv2.waitKey(0)
    cv2.destroyWindow("Wilson Calibration - Quadrant Histograms")


def _show_depth_histogram(
    frame_stack: np.ndarray,
    dsurface_map: np.ndarray,
    percentile: float,
    surface_offset: int,
) -> None:
    """Display 4 histograms (one per quadrant) for calibration debugging.

    Shows for each quadrant:
    - Distribution of depth readings
    - Percentile markers (p50, p90, p95, p97.5, p99)
    - dsurface and dmax lines
    """
    # frame_stack shape: (num_frames, height, width)
    _, height, width = frame_stack.shape
    mid_h, mid_w = height // 2, width // 2

    # Define quadrants: (name, row_slice, col_slice)
    quadrants = [
        ("Top-Left", slice(0, mid_h), slice(0, mid_w)),
        ("Top-Right", slice(0, mid_h), slice(mid_w, width)),
        ("Bottom-Left", slice(mid_h, height), slice(0, mid_w)),
        ("Bottom-Right", slice(mid_h, height), slice(mid_w, width)),
    ]

    # Single histogram dimensions
    hist_w, hist_h = 400, 200
    padding = 10

    # Create combined image (2x2 grid)
    img_width = hist_w * 2 + padding * 3
    img_height = (hist_h + 60) * 2 + padding * 3
    combined_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Get global depth range for consistent x-axis
    all_valid = frame_stack[~np.isnan(frame_stack)]
    if len(all_valid) == 0:
        print("[Debug] No valid depth values")
        return
    global_min, global_max = np.min(all_valid), np.max(all_valid)
    global_range = global_max - global_min

    if global_range < 1:
        print(f"[Debug] Depth range too small: {global_min:.0f} - {global_max:.0f}")
        return

    # Percentile colors
    pcolors = {
        50: (255, 255, 0),    # Cyan
        90: (255, 150, 0),    # Blue
        95: (0, 255, 0),      # Green
        97.5: (0, 255, 255),  # Yellow
        99: (0, 150, 255),    # Orange
    }

    for idx, (name, row_slice, col_slice) in enumerate(quadrants):
        # Extract quadrant data
        quad_data = frame_stack[:, row_slice, col_slice]
        quad_dsurface = dsurface_map[row_slice, col_slice]

        valid_depths = quad_data[~np.isnan(quad_data)].flatten()
        if len(valid_depths) == 0:
            continue

        # Compute percentiles
        pvals = {p: np.percentile(valid_depths, p) for p in [50, 90, 95, 97.5, 99]}

        # dsurface and dmax for this quadrant
        valid_ds = quad_dsurface[quad_dsurface > 0]
        dsurface_mean = np.mean(valid_ds) if len(valid_ds) > 0 else 0
        dmax_mean = dsurface_mean - surface_offset

        # Create single histogram
        hist_img = np.zeros((hist_h + 60, hist_w, 3), dtype=np.uint8)

        # Compute histogram with global range
        num_bins = 100
        hist, _ = np.histogram(valid_depths, bins=num_bins, range=(global_min, global_max))
        hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist

        # Draw bars
        bar_w = hist_w // num_bins
        for i, h in enumerate(hist_norm):
            x1, x2 = i * bar_w, (i + 1) * bar_w - 1
            y1 = int(hist_h * (1 - h))
            cv2.rectangle(hist_img, (x1, y1), (x2, hist_h), (80, 80, 80), -1)

        # Helper for x coordinate
        def d2x(d: float) -> int:
            return int((d - global_min) / global_range * hist_w)

        # Draw percentile lines
        for p, val in pvals.items():
            x = d2x(val)
            color = pcolors.get(p, (200, 200, 200))
            cv2.line(hist_img, (x, 0), (x, hist_h), color, 1)

        # Draw dsurface line (yellow, thick)
        dsurface_x = d2x(dsurface_mean)
        cv2.line(hist_img, (dsurface_x, 0), (dsurface_x, hist_h), (0, 255, 255), 2)

        # Draw dmax line (red, thick)
        dmax_x = d2x(dmax_mean)
        cv2.line(hist_img, (dmax_x, 0), (dmax_x, hist_h), (0, 0, 255), 2)

        # Title and info
        cv2.putText(hist_img, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(hist_img, f"dsurface: {dsurface_mean:.0f}",
                    (5, hist_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(hist_img, f"dmax: {dmax_mean:.0f}",
                    (5, hist_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(hist_img, f"n={len(valid_depths)//1000}k",
                    (5, hist_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Place in combined image
        row, col = idx // 2, idx % 2
        y_off = padding + row * (hist_h + 60 + padding)
        x_off = padding + col * (hist_w + padding)
        combined_img[y_off:y_off + hist_h + 60, x_off:x_off + hist_w] = hist_img

    # Add legend at bottom
    legend_y = img_height - 25
    cv2.putText(combined_img, f"Range: {global_min:.0f}-{global_max:.0f}mm | ",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    legend_x = 200
    for p, color in pcolors.items():
        cv2.putText(combined_img, f"p{p}", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        legend_x += 45
    cv2.putText(combined_img, "dsurface", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    legend_x += 60
    cv2.putText(combined_img, "dmax", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Show
    cv2.namedWindow("Calibration - Quadrant Histograms", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibration - Quadrant Histograms", combined_img)
    print("[Debug] Showing quadrant histograms. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration - Quadrant Histograms")


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
