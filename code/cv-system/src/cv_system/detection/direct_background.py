"""
Background model for DIRECT touch detector.

Implements a per-pixel rolling mean/stddev using exponential moving average,
approximating the original DIRECT's sliding window approach from
BackgroundUpdaterThread.cpp.
"""

from __future__ import annotations

import numpy as np


class DIRECTBackgroundModel:
    """
    Rolling background model matching DIRECT algorithm.

    Uses Exponential Moving Average (EMA) instead of sliding window for efficiency.
    EMA with alpha=0.02 approximates a 100-frame sliding window.
    """

    # Matching original DIRECT parameters
    MIN_DEPTH = 100  # mm - ignore values below
    MAX_DEPTH = 50000  # mm - ignore values above
    MIN_FRAMES = 30  # frames before considering stable

    # EMA parameter (alpha = 2/(N+1) for N=100 frame window)
    ALPHA = 0.02

    # Heuristics from original DIRECT code
    HEUR_Z_INCREASE_THRESHOLD = 10.0  # z-scores to mark unstable
    HEUR_STABLE_FACTOR = 1.0  # mm / m^2 - stability threshold
    HEUR_HALO_THRESHOLD = 5.0  # mm - reject halos (multipath interference)

    def __init__(self, shape: tuple[int, int]) -> None:
        """
        Initialize background model from scratch.

        Args:
            shape: (height, width) of depth frames
        """
        h, w = shape
        self._shape = shape

        # EMA statistics - start uninitialized
        self._mean = np.zeros((h, w), dtype=np.float32)
        self._m2 = np.zeros((h, w), dtype=np.float32)  # For Welford's algorithm
        self._frame_count = np.zeros((h, w), dtype=np.int32)
        self._initialized = np.zeros((h, w), dtype=bool)

        # Stable values (updated only when pixel is stable)
        self._stable_mean = np.zeros((h, w), dtype=np.float32)
        self._stable_stddev = np.full((h, w), 1e6, dtype=np.float32)
        self._stable = np.zeros((h, w), dtype=bool)

    def update(self, depth_frame: np.ndarray) -> None:
        """
        Update background model with new depth frame.

        Uses Welford's online algorithm for numerically stable mean/variance.

        Args:
            depth_frame: Raw depth frame (uint16, mm values)
        """
        valid = (depth_frame >= self.MIN_DEPTH) & (depth_frame <= self.MAX_DEPTH)
        depth_float = depth_frame.astype(np.float32)

        # First valid value initializes the pixel
        first_time = valid & ~self._initialized
        self._mean[first_time] = depth_float[first_time]
        self._m2[first_time] = 0
        self._initialized[first_time] = True

        # Already initialized pixels: use Welford's algorithm
        update_mask = valid & self._initialized & ~first_time

        # Increment frame count (cap at 90 to avoid overflow)
        self._frame_count[valid] = np.minimum(
            self._frame_count[valid] + 1, self.MIN_FRAMES * 3
        )

        # EMA update for mean and variance
        if np.any(update_mask):
            delta = depth_float[update_mask] - self._mean[update_mask]

            # EMA update for mean
            self._mean[update_mask] += self.ALPHA * delta

            # EMA update for variance: var = (1-α) * (var + α * δ²)
            self._m2[update_mask] = (1 - self.ALPHA) * (
                self._m2[update_mask] + self.ALPHA * delta * delta
            )

        # Update stability and stable values
        self._update_stability()

    def _update_stability(self) -> None:
        """Update stability flags and stable mean/stddev."""
        # Compute stddev from M2 (EMA of squared deviations)
        stddev = np.sqrt(np.maximum(self._m2, 1e-6))

        # Only consider initialized pixels
        initialized = self._initialized

        # Heuristic 1: Z-increase destabilization
        # Only destabilize when mean INCREASES (depth got farther = object removed)
        # This matches original DIRECT: if(cur_mean > stable_mean + stable_stdev * 10)
        safe_stable_stddev = np.maximum(self._stable_stddev, 1.0)  # At least 1mm
        mean_increase = self._mean - self._stable_mean
        destabilize = initialized & (mean_increase > safe_stable_stddev * self.HEUR_Z_INCREASE_THRESHOLD)

        # Heuristic 2: Noise check
        # stddev > FACTOR * (depth/1000)^2 means pixel is too noisy
        depth_m = self._mean / 1000.0
        max_stddev = self.HEUR_STABLE_FACTOR * depth_m * depth_m
        # Use a minimum threshold of 5mm to be more lenient during convergence
        max_stddev = np.maximum(max_stddev, 5.0)
        too_noisy = initialized & (stddev > max_stddev)

        # Not enough frames yet
        not_enough = self._frame_count < self.MIN_FRAMES

        # Update stability: must be initialized, have enough frames, not destabilized, not too noisy
        self._stable = initialized & ~destabilize & ~too_noisy & ~not_enough

        # Reset stable values where destabilized
        self._stable_mean[destabilize] = 0
        self._stable_stddev[destabilize] = 1e6

        # Heuristic 3: Update stable values where stable
        # For newly stable pixels (stable_mean == 0), always update
        # For already stable pixels, only update if mean changed significantly
        newly_stable = self._stable & (self._stable_mean == 0)
        update_existing = self._stable & (self._stable_mean > 0) & (
            (self._mean > self._stable_mean + self.HEUR_HALO_THRESHOLD)
            | (self._mean < self._stable_mean)
        )
        update_mask = newly_stable | update_existing

        self._stable_mean[update_mask] = self._mean[update_mask]
        # Use at least 2mm stddev to avoid overly sensitive destabilization
        self._stable_stddev[update_mask] = np.maximum(stddev[update_mask], 2.0)

    @property
    def mean(self) -> np.ndarray:
        """Get stable background mean."""
        return self._stable_mean

    @property
    def stddev(self) -> np.ndarray:
        """Get stable background stddev."""
        # Return stable stddev, but use a reasonable minimum for detection
        return np.maximum(self._stable_stddev, 2.0)  # At least 2mm

    @property
    def stable_mask(self) -> np.ndarray:
        """Get boolean mask of stable pixels."""
        return self._stable

    @property
    def stable_percentage(self) -> float:
        """Get percentage of stable pixels (0-100)."""
        return float(np.mean(self._stable) * 100)

    @property
    def is_ready(self) -> bool:
        """Check if enough pixels are stable for detection."""
        return np.mean(self._stable) > 0.5  # >50% pixels stable
