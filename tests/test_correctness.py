from fractions import Fraction
import io
import json
from collections import defaultdict

from video_indexer.aggregate import SecondAggregator
from video_indexer.pipeline import video_records


def aggregate(counts):
    agg = SecondAggregator("x.parquet:0")
    for count in counts:
        list(agg.add(0, count))
    return list(agg.finish())[0]


def test_exactly_half_is_present_and_counts_are_per_frame():
    result = aggregate([(3, 1, 0, 0, 0), (0, 0, 0, 0, 0)])
    assert result["objects"] == ["person", "car"]
    assert result["people"] == 1  # floor((0 + 3) / 2)


def test_median_is_neither_mean_nor_max():
    result = aggregate([(n, 0, 0, 0, 0) for n in [10, 0, 2, 2, 2]])
    assert result["people"] == 2
    assert result["objects"] == ["person"]


def test_timestamp_gaps_and_last_partial_second():
    agg = SecondAggregator("v")
    assert list(agg.add(0, (0,) * 5)) == []
    records = list(agg.add(3, (1, 0, 0, 0, 0))) + list(agg.finish())
    assert [r["second"] for r in records] == [0, 1, 2, 3]
    assert records[1]["objects"] == []
    assert records[-1]["people"] == 1


def test_decode_failure_preserves_completed_seconds_and_pending_batch():
    def frames():
        yield 0, None
        yield 0, None
        yield 1, None
        raise ValueError("damaged")
    result = list(video_records("v", frames(), lambda batch: [(1, 0, 0, 0, 0)] * len(batch), 8))
    assert result[0] == {"video": "v", "second": 0, "objects": ["person"], "people": 1}
    assert len(result) == 2 and "error" in result[-1]


def test_empty_video_is_an_error():
    assert "error" in list(video_records("v", iter(()), lambda x: [], 1))[0]


def test_batching_does_not_change_temporal_aggregation():
    frames = [(i // 7, i) for i in range(23)]
    detector = lambda batch: [(i % 4, int(i % 3 == 0), 0, 0, 0) for i in batch]
    reference = list(video_records("v", iter(frames), detector, 1))
    assert list(video_records("v", iter(frames), detector, 8)) == reference


def test_rational_timestamps_do_not_round_across_second_boundary():
    from video_indexer.decode import TimestampClock
    clock = TimestampClock(Fraction(30))
    assert clock.second(90000, Fraction(1, 90000)) == 0
    assert clock.second(179999, Fraction(1, 90000)) == 0
    assert clock.second(180000, Fraction(1, 90000)) == 1
    assert clock.second(270000, Fraction(1, 90000)) == 2


def test_rotation_preserves_bgr_channels():
    import numpy as np
    from video_indexer.decode import rotate_bgr
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(rotate_bgr(image, 90), np.rot90(image))
    assert rotate_bgr(image, 90).flags.c_contiguous


def test_variable_timestamps_are_not_replaced_with_average_fps():
    from video_indexer.decode import TimestampClock
    clock = TimestampClock(Fraction(30))
    assert [clock.second(pts, Fraction(1, 1000)) for pts in (5000, 5010, 5999, 6000, 8250)] == [0, 0, 0, 1, 3]


def test_display_rotation_matches_ffmpeg(tmp_path):
    import subprocess
    import av
    import numpy as np
    from video_indexer.decode import decoded_frames
    original, rotated = tmp_path / "original.mp4", tmp_path / "rotated.mp4"
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :24] = (0, 0, 255)
    image[:, 24:] = (255, 0, 0)
    with av.open(str(original), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=1)
        stream.width, stream.height, stream.pix_fmt = 48, 32, "yuv420p"
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in list(stream.encode(frame)) + list(stream.encode()):
            output.mux(packet)
    # Recent FFmpeg versions no longer translate output rotate metadata into a
    # display matrix. Older Ubuntu FFmpeg uses the legacy metadata spelling.
    result = subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(original), "-c", "copy", str(rotated)], capture_output=True)
    if result.returncode:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(rotated)], check=True)
    ours = next(decoded_frames(rotated))[1]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(rotated), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    expected = np.frombuffer(raw, dtype=np.uint8).reshape(48, 32, 3)
    assert ours.shape == expected.shape
    assert np.max(np.abs(ours.astype(int) - expected.astype(int))) <= 3


def test_valid_and_corrupt_video_rows(tmp_path):
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from video_indexer.pipeline import process_parquet
    movie = tmp_path / "fixture.mp4"
    with av.open(str(movie), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for _ in range(6):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    shard = tmp_path / "train-00000-of-01357.parquet"
    pq.write_table(pa.table({"mp4": [movie.read_bytes(), b"not a video"]}), shard)
    destination = io.StringIO()
    metrics = process_parquet(shard, destination, lambda batch: [(0,) * 5] * len(batch), batch_size=4)
    records = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert [r["second"] for r in records[:-1]] == [0, 1]
    assert all(r["objects"] == [] and r["people"] == 0 for r in records[:-1])
    assert records[-1]["video"] == shard.name + ":1" and "error" in records[-1]
    assert metrics["frames"] == 6 and metrics["videos"] == 2
