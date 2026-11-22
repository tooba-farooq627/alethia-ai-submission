import pytest
import torch

from video_indexer.pipeline import Detector


def test_oom_retry_preserves_every_frame_and_order():
    detector = Detector.__new__(Detector)
    detector.frames = 0
    calls = []

    def counts(images):
        calls.append(list(images))
        if len(images) > 2:
            raise torch.cuda.OutOfMemoryError("synthetic memory limit")
        return [(image, 0, 0, 0, 0) for image in images]

    detector._counts = counts
    assert detector(list(range(7))) == [(i, 0, 0, 0, 0) for i in range(7)]
    assert detector.frames == 7
    assert calls[0] == list(range(7))


def test_single_frame_oom_is_not_silently_dropped():
    detector = Detector.__new__(Detector)
    detector.frames = 0

    def counts(images):
        raise torch.cuda.OutOfMemoryError("synthetic single-frame failure")

    detector._counts = counts
    with pytest.raises(torch.cuda.OutOfMemoryError):
        detector([0])
    assert detector.frames == 0
