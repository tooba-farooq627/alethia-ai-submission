"""Decode frames using presentation timestamps, not frame-index / FPS."""
import json
import math
import subprocess
from fractions import Fraction

import av
import numpy as np


class TimestampClock:
    def __init__(self, rate=None):
        self.step = Fraction(1, 1) / rate if rate and rate > 0 else None
        self.origin = None
        self.previous = None

    def second(self, pts, time_base):
        stamp = Fraction(pts) * time_base if pts is not None and time_base else None
        if self.previous is None:
            self.origin = stamp
            elapsed = Fraction(0)
        elif stamp is not None:
            if self.origin is None:
                if self.step is None:
                    raise ValueError("cannot reconstruct missing initial timestamp")
                self.origin = stamp - (self.previous + self.step)
            elapsed = stamp - self.origin
        elif self.step is not None:
            elapsed = self.previous + self.step
        else:
            raise ValueError("frame has no timestamp or usable frame rate")
        if elapsed < 0 or (self.previous is not None and elapsed < self.previous):
            raise ValueError("non-monotonic presentation timestamps")
        self.previous = elapsed
        return elapsed.numerator // elapsed.denominator


def rotation(path):
    # ffprobe reads MP4 display matrices even when the stream's rotate tag is absent.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise ValueError("video has no video stream")
    stream = streams[0]
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            # ffprobe display-matrix angles are counterclockwise.
            return float(side["rotation"])
    # Legacy rotate metadata is a clockwise display instruction.
    return -float(stream.get("tags", {}).get("rotate", 0))


def rotate_bgr(image, angle):
    if not math.isfinite(angle):
        raise ValueError("invalid rotation metadata")
    quarter_turns = angle / 90
    if not math.isclose(quarter_turns, round(quarter_turns), abs_tol=0.001):
        import cv2
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = math.ceil(height * sine + width * cosine)
        new_height = math.ceil(height * cosine + width * sine)
        matrix[0, 2] += new_width / 2 - width / 2
        matrix[1, 2] += new_height / 2 - height / 2
        return cv2.warpAffine(image, matrix, (new_width, new_height))
    return np.ascontiguousarray(np.rot90(image, round(quarter_turns) % 4))


def decoded_frames(path, threads=2):
    angle = rotation(path)
    # PyAV does not apply the container display rotation itself.
    with av.open(str(path), options={"err_detect": "explode"}) as container:
        if not container.streams.video:
            raise ValueError("video has no video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        stream.thread_type = "FRAME"
        clock = TimestampClock(stream.average_rate)
        for frame in container.decode(stream):
            if frame.is_corrupt:
                raise ValueError("corrupt video frame")
            second = clock.second(frame.pts, frame.time_base)
            image = frame.to_ndarray(format="bgr24")
            yield second, rotate_bgr(image, angle) if angle else image
