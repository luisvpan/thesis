"""
Depth reconstruction using IR as guide with Guided Filter.

This example demonstrates how to use the infrared (IR) frame from the Kinect
to reconstruct missing depth data, particularly useful for recovering
fingertip depth values that are often lost due to sensor limitations.

The algorithm uses OpenCV's Guided Filter (cv2.ximgproc.guidedFilter) to
propagate depth values into holes/invalid regions using the IR frame as a guide.
The IR frame provides edge information that tells the filter where the physical
boundaries of objects (like fingers) are located.

Requirements:
    - Kinect for Windows v2 SDK installed
    - Kinect v2 sensor connected
    - opencv-python installed
    - opencv-contrib-python installed (for cv2.ximgproc)

Usage:
    uv run python examples/depth_ir_reconstruction.py

Controls:
    - 'r': Toggle radius value (3, 5, 7, 9)
    - 'e': Toggle eps value (0.001, 0.01, 0.1, 0.5)
    - 'm': Toggle morphology kernel size (3, 5, 7)
    - 'd': Toggle hand detection depth range
    - 's': Save current frames to disk
    - 'q': Quit
"""

import cv2
import numpy as np
import time
import os
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


def check_ximgproc():
    """Check if opencv-contrib-python is installed with ximgproc module."""
    try:
        _ = cv2.ximgproc.guidedFilter
        return True
    except AttributeError:
        print("ERROR: cv2.ximgproc not found!")
        print("Please install opencv-contrib-python:")
        print("  uv pip install opencv-contrib-python")
        return False


def normalize_depth_for_display(depth_frame, max_depth=4500):
    """
    Normalize depth frame to 8-bit for display.

    Args:
        depth_frame: Raw depth frame (uint16, values in mm)
        max_depth: Maximum depth value for normalization

    Returns:
        depth_8bit: Normalized 8-bit depth image
    """
    depth_normalized = (depth_frame.astype(np.float32) / max_depth * 255)
    depth_8bit = np.clip(depth_normalized, 0, 255).astype(np.uint8)
    return depth_8bit


def normalize_ir_for_display(ir_frame, clip_max=4000):
    """
    Normalize IR frame to 8-bit for display.

    Args:
        ir_frame: Raw IR frame (uint16)
        clip_max: Maximum value to clip before normalization

    Returns:
        ir_8bit: Normalized 8-bit IR image
    """
    ir_clipped = np.clip(ir_frame, 1, clip_max)
    ir_8bit = (ir_clipped / clip_max * 255).astype(np.uint8)
    return ir_8bit


def reconstruct_depth_with_ir(depth_raw, ir_8bit, radius=5, eps=0.01, morph_kernel_size=3):
    """
    Reconstruct depth frame using IR as guide with Guided Filter.

    This algorithm:
    1. Identifies invalid (zero) depth regions (holes at fingertips)
    2. Cleans noisy edges using morphological operations
    3. Uses Guided Filter to propagate depth using IR as guide
    4. Replaces only the hole regions with the filtered result

    Args:
        depth_raw: Raw depth frame (uint16, 512x424)
        ir_8bit: IR frame normalized to 8-bit (uint8, 512x424)
        radius: Guided filter radius (larger = more smoothing)
        eps: Guided filter regularization (smaller = more edge-preserving)
        morph_kernel_size: Morphology kernel size for cleaning

    Returns:
        depth_final: Reconstructed depth frame
        mask_invalid: Mask of originally invalid pixels
        mask_cleaned: Cleaned mask after morphology
    """
    # 1. Identify invalid (zero) depth regions
    mask_invalid = (depth_raw == 0).astype(np.uint8) * 255

    # 2. Clean noisy edges using morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_cleaned = cv2.morphologyEx(mask_invalid, cv2.MORPH_CLOSE, kernel)

    # 3. Convert depth to float32 for guided filter (required by ximgproc)
    depth_float = depth_raw.astype(np.float32)

    # 4. Apply Guided Filter using IR as guide
    # The IR image provides edge information that tells the filter
    # where the physical boundaries of objects are
    depth_refined = cv2.ximgproc.guidedFilter(
        guide=ir_8bit,
        src=depth_float,
        radius=radius,
        eps=eps * (255**2)  # Scale eps relative to guide range
    )

    # 5. Replace only the hole regions with filtered result
    depth_final = np.where(mask_cleaned > 0, depth_refined, depth_raw)
    depth_final = depth_final.astype(np.uint16)

    return depth_final, mask_invalid, mask_cleaned


def detect_hand_from_depth(depth_frame, min_depth=300, max_depth=1000, min_area=3000):
    """
    Detect hand using depth thresholding (objects close to camera).

    Args:
        depth_frame: Raw depth frame (uint16)
        min_depth: Minimum depth in mm (filter noise)
        max_depth: Maximum depth in mm (hand range)
        min_area: Minimum contour area to be considered a hand

    Returns:
        contour: Hand contour or None
        fingertips: List of (x, y) fingertip positions
        center: (cx, cy) center of hand or None
    """
    # Create mask for objects within depth range
    hand_mask = ((depth_frame > min_depth) & (depth_frame < max_depth)).astype(np.uint8) * 255

    # Clean up noise with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, [], None

    # Get largest contour (assumed to be hand)
    hand_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(hand_contour) < min_area:
        return None, [], None

    # Calculate center
    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return hand_contour, [], None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Find fingertips using convex hull defects
    fingertips = []
    try:
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(hand_contour[s][0])
                if d > 10000:  # Significant defect
                    fingertips.append(start)

        # If not enough fingertips from defects, use hull points
        if len(fingertips) < 3:
            hull_points = cv2.convexHull(hand_contour, returnPoints=True)
            candidates = []
            for point in hull_points:
                x, y = point[0]
                dist = (x - cx) ** 2 + (y - cy) ** 2
                candidates.append((x, y, dist))
            candidates.sort(key=lambda p: -p[2])
            fingertips = [(p[0], p[1]) for p in candidates[:5]]
    except Exception:
        pass

    return hand_contour, fingertips[:5], (cx, cy)


def draw_hand_overlay(image, contour, fingertips, center, color=(0, 255, 0)):
    """Draw hand contour, fingertips and center on image."""
    if contour is not None:
        cv2.drawContours(image, [contour], -1, color, 2)

    if center is not None:
        cv2.circle(image, center, 8, (255, 255, 0), -1)
        cv2.circle(image, center, 10, (255, 255, 255), 2)

    for tip in fingertips:
        cv2.circle(image, tip, 6, (0, 0, 255), -1)
        cv2.circle(image, tip, 8, (255, 255, 255), 2)

    return image


def create_comparison_view(depth_original, depth_reconstructed, ir_8bit, mask, hand_data=None):
    """
    Create a side-by-side comparison view with color mapping.

    Args:
        depth_original: Original depth frame
        depth_reconstructed: Reconstructed depth frame
        ir_8bit: IR frame (8-bit)
        mask: Invalid pixel mask
        hand_data: Tuple of (contour, fingertips, center) or None

    Returns:
        comparison: Combined visualization image
    """
    # Normalize depth for display
    depth_orig_8bit = normalize_depth_for_display(depth_original)
    depth_recon_8bit = normalize_depth_for_display(depth_reconstructed)

    # Apply colormap to depth
    depth_orig_color = cv2.applyColorMap(depth_orig_8bit, cv2.COLORMAP_JET)
    depth_recon_color = cv2.applyColorMap(depth_recon_8bit, cv2.COLORMAP_JET)

    # Mark invalid regions in original (red overlay)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    depth_orig_marked = cv2.addWeighted(depth_orig_color, 0.7, mask_3ch, 0.3, 0)

    # Create IR with 3 channels
    ir_3ch = cv2.cvtColor(ir_8bit, cv2.COLOR_GRAY2BGR)

    # Create difference view (highlight reconstructed regions)
    diff = np.abs(depth_recon_8bit.astype(np.int16) - depth_orig_8bit.astype(np.int16))
    diff_normalized = np.clip(diff * 5, 0, 255).astype(np.uint8)  # Amplify differences
    diff_color = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_HOT)

    # Draw hand overlay on all views
    if hand_data is not None:
        contour, fingertips, center = hand_data
        depth_orig_marked = draw_hand_overlay(depth_orig_marked, contour, fingertips, center, (0, 255, 0))
        ir_3ch = draw_hand_overlay(ir_3ch, contour, fingertips, center, (0, 255, 0))
        depth_recon_color = draw_hand_overlay(depth_recon_color, contour, fingertips, center, (0, 255, 0))
        diff_color = draw_hand_overlay(diff_color, contour, fingertips, center, (0, 255, 0))

    # Combine into 2x2 grid
    top_row = np.hstack([depth_orig_marked, ir_3ch])
    bottom_row = np.hstack([depth_recon_color, diff_color])
    comparison = np.vstack([top_row, bottom_row])

    return comparison


def add_text_overlay(image, params, stats):
    """Add parameter and statistics overlay to image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Labels for quadrants
    labels = [
        ("Original Depth (holes in red)", (10, 20)),
        ("IR Frame (guide)", (522, 20)),
        ("Reconstructed Depth", (10, 444)),
        ("Difference (reconstructed regions)", (522, 444)),
    ]

    for text, pos in labels:
        cv2.putText(image, text, pos, font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(image, text, pos, font, font_scale, (255, 255, 255), thickness)

    # Parameters
    param_text = f"Radius: {params['radius']} | Eps: {params['eps']:.3f} | Morph: {params['morph']} | Hand: {params['hand_min']}-{params['hand_max']}mm"
    cv2.putText(image, param_text, (10, 830), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, param_text, (10, 830), font, font_scale, (0, 255, 0), thickness)

    # Statistics
    hand_text = f"Hand: {'Detected' if stats['hand_detected'] else 'Not found'}"
    stats_text = f"Invalid: {stats['invalid_count']} ({stats['invalid_pct']:.2f}%) | Recovered: {stats['recovered_count']} | {hand_text}"
    cv2.putText(image, stats_text, (10, 850), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, stats_text, (10, 850), font, font_scale, (255, 255, 0), thickness)

    # Controls
    controls = "Controls: [R]adius [E]ps [M]orph [D]epth range [S]ave [Q]uit"
    cv2.putText(image, controls, (300, 830), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, controls, (300, 830), font, font_scale, (200, 200, 200), thickness)

    return image


def main():
    if not check_ximgproc():
        return

    print("Initializing Kinect with Depth + IR sources...")

    # Initialize Kinect with Depth and Infrared sources
    try:
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Depth |
            PyKinectV2.FrameSourceTypes_Infrared
        )
    except RuntimeError as e:
        print(f"Failed to initialize Kinect: {e}")
        return

    print("Kinect initialized successfully!")
    print(f"Depth resolution: {kinect.depth_frame_desc.Width}x{kinect.depth_frame_desc.Height}")
    print(f"IR resolution: {kinect.infrared_frame_desc.Width}x{kinect.infrared_frame_desc.Height}")
    print()
    print("Controls:")
    print("  [R] - Cycle radius values (3, 5, 7, 9)")
    print("  [E] - Cycle eps values (0.001, 0.01, 0.1, 0.5)")
    print("  [M] - Cycle morphology kernel size (3, 5, 7)")
    print("  [D] - Cycle hand detection depth range")
    print("  [S] - Save current frames to disk")
    print("  [Q] - Quit")
    print()

    # Parameters with cycling options
    radius_options = [3, 5, 7, 9]
    eps_options = [0.001, 0.01, 0.1, 0.5]
    morph_options = [3, 5, 7]
    # Hand detection depth ranges (min_depth, max_depth) in mm
    hand_depth_options = [
        (300, 800),    # Very close
        (300, 1000),   # Close (default)
        (300, 1500),   # Medium
        (500, 2000),   # Far
    ]

    radius_idx = 1  # Start with radius=5
    eps_idx = 1     # Start with eps=0.01
    morph_idx = 0   # Start with morph=3
    hand_depth_idx = 1  # Start with 300-1000mm

    last_depth_frame = None
    last_ir_frame = None
    frame_count = 0
    save_count = 0

    # Create output directory for saved frames
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "depth_ir_output")

    cv2.namedWindow("Depth IR Reconstruction", cv2.WINDOW_NORMAL)

    while True:
        # Get current parameters
        radius = radius_options[radius_idx]
        eps = eps_options[eps_idx]
        morph = morph_options[morph_idx]
        hand_min, hand_max = hand_depth_options[hand_depth_idx]

        # Get depth frame
        if kinect.has_new_depth_frame():
            depth_data = kinect.get_last_depth_frame()
            if depth_data is not None:
                last_depth_frame = depth_data.reshape((424, 512))

        # Get IR frame
        if kinect.has_new_infrared_frame():
            ir_data = kinect.get_last_infrared_frame()
            if ir_data is not None:
                last_ir_frame = ir_data.reshape((424, 512))

        # Process and display when we have both frames
        if last_depth_frame is not None and last_ir_frame is not None:
            # Normalize IR to 8-bit
            ir_8bit = normalize_ir_for_display(last_ir_frame)

            # Reconstruct depth using IR as guide
            depth_reconstructed, mask_invalid, mask_cleaned = reconstruct_depth_with_ir(
                last_depth_frame, ir_8bit, radius, eps, morph
            )

            # Detect hand from reconstructed depth
            hand_contour, fingertips, hand_center = detect_hand_from_depth(
                depth_reconstructed, min_depth=hand_min, max_depth=hand_max
            )
            hand_data = (hand_contour, fingertips, hand_center) if hand_contour is not None else None

            # Calculate statistics
            total_pixels = last_depth_frame.size
            invalid_count = np.sum(mask_invalid > 0)
            invalid_pct = (invalid_count / total_pixels) * 100

            # Count recovered pixels (pixels that had no depth but now have valid depth)
            recovered_mask = (mask_cleaned > 0) & (depth_reconstructed > 0)
            recovered_count = np.sum(recovered_mask)

            stats = {
                'invalid_count': invalid_count,
                'invalid_pct': invalid_pct,
                'recovered_count': recovered_count,
                'hand_detected': hand_contour is not None
            }

            params = {
                'radius': radius,
                'eps': eps,
                'morph': morph,
                'hand_min': hand_min,
                'hand_max': hand_max
            }

            # Create comparison visualization with hand overlay
            comparison = create_comparison_view(
                last_depth_frame, depth_reconstructed, ir_8bit, mask_cleaned, hand_data
            )

            # Add text overlay
            comparison = add_text_overlay(comparison, params, stats)

            cv2.imshow("Depth IR Reconstruction", comparison)
            frame_count += 1

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            radius_idx = (radius_idx + 1) % len(radius_options)
            print(f"Radius changed to: {radius_options[radius_idx]}")
        elif key == ord('e'):
            eps_idx = (eps_idx + 1) % len(eps_options)
            print(f"Eps changed to: {eps_options[eps_idx]}")
        elif key == ord('m'):
            morph_idx = (morph_idx + 1) % len(morph_options)
            print(f"Morphology kernel changed to: {morph_options[morph_idx]}")
        elif key == ord('d'):
            hand_depth_idx = (hand_depth_idx + 1) % len(hand_depth_options)
            new_min, new_max = hand_depth_options[hand_depth_idx]
            print(f"Hand depth range changed to: {new_min}-{new_max}mm")
        elif key == ord('s'):
            if last_depth_frame is not None and last_ir_frame is not None:
                os.makedirs(output_dir, exist_ok=True)

                # Save raw frames
                depth_path = os.path.join(output_dir, f"depth_raw_{save_count}.npy")
                ir_path = os.path.join(output_dir, f"ir_raw_{save_count}.npy")
                recon_path = os.path.join(output_dir, f"depth_reconstructed_{save_count}.npy")

                np.save(depth_path, last_depth_frame)
                np.save(ir_path, last_ir_frame)
                np.save(recon_path, depth_reconstructed)

                # Save visualization
                vis_path = os.path.join(output_dir, f"comparison_{save_count}.png")
                cv2.imwrite(vis_path, comparison)

                print(f"Saved frames to {output_dir} (set {save_count})")
                save_count += 1

    # Cleanup
    kinect.close()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")
    print("Kinect closed.")


if __name__ == "__main__":
    main()
