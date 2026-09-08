from cv_system.config import CameraConfig

class ResolutionMapper:
    """Maps pixel coordinates between RGB and depth frames.
    
    With OpenNI2's IMAGE_REGISTRATION_DEPTH_TO_COLOR enabled, both frames
    share the same coordinate system — the mapping is a pure resolution
    scaling operation.
    """

    def __init__(self, config: CameraConfig) -> None:
        # (height, width) convention, consistent with CameraConfig
        self._rgb_h, self._rgb_w = config.rgb_resolution
        self._depth_h, self._depth_w = config.depth_resolution

    def rgb_to_depth(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [
            (int(x * self._depth_w / self._rgb_w), int(y * self._depth_h / self._rgb_h))
            for x, y in points
        ]

    def depth_to_rgb(self, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [
            (int(x * self._rgb_w / self._depth_w), int(y * self._rgb_h / self._depth_h))
            for x, y in points
        ]