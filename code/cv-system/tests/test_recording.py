import json

import cv2
import numpy as np

from cv_system.recording import SessionRecorder, finish_manifest, write_session_manifest


def test_recorder_keeps_video_and_log_aligned(tmp_path):
    recorder = SessionRecorder(tmp_path, "s", fps=30.0)
    for i in range(25):
        recorder.write(np.full((48, 64, 3), i * 10 % 255, np.uint8), [])
    recorder.close()

    video = cv2.VideoCapture(str(recorder.video_path))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 25
    lines = recorder.log_path.read_text().splitlines()
    assert [json.loads(line)["frame"] for line in lines] == list(range(25))
    assert recorder.effective_fps > 0

    manifest = tmp_path / "s.json"
    write_session_manifest(manifest, {"session_id": "s"})
    finish_manifest(manifest, recorder, 30.0)
    data = json.loads(manifest.read_text())
    assert data["video_frames"] == 25
    assert data["video_fps_nominal"] == 30.0
    assert "effective_fps" in data and "encoder_waits" in data
