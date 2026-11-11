import math

from cv_system.hardware.frame_clock import FrameClock


def test_frame_clock_counts_new_frames_and_age():
    clock = FrameClock()
    assert clock.index == 0
    assert math.isinf(clock.age_s)
    clock.tick()
    clock.tick()
    assert clock.index == 2
    assert 0 <= clock.age_s < 1
