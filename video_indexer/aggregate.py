"""Exact aggregation, independent of decoding and model libraries."""
from collections import Counter
from dataclasses import dataclass, field

OBJECTS = ("person", "car", "dog", "laptop", "cell phone")
CLASS_IDS = (0, 2, 16, 63, 67)  # Official COCO class indices.


@dataclass
class SecondAggregator:
    video: str
    second: int = 0
    frames: int = 0
    presence: list[int] = field(default_factory=lambda: [0] * 5)
    people_histogram: Counter = field(default_factory=Counter)

    def _record(self):
        people = 0
        if self.frames:
            # Zero-based central indices; average and floor for an even sample.
            low, high = (self.frames - 1) // 2, self.frames // 2
            cumulative, left = 0, None
            for count, frequency in sorted(self.people_histogram.items()):
                cumulative += frequency
                if left is None and cumulative > low:
                    left = count
                if cumulative > high:
                    people = (left + count) // 2
                    break
        return {"video": self.video, "second": self.second,
                "objects": [name for name, hits in zip(OBJECTS, self.presence)
                            if self.frames and 2 * hits >= self.frames],
                "people": people}

    def add(self, second, counts):
        if second < self.second:
            raise ValueError("frame timestamps moved backwards")
        if len(counts) != 5 or any(c < 0 for c in counts):
            raise ValueError("invalid per-frame counts")
        # Empty timestamp gaps still receive a record.
        while self.second < second:
            yield self._record()
            self.second += 1
            self.frames = 0
            self.presence = [0] * 5
            self.people_histogram.clear()
        self.frames += 1
        self.people_histogram[counts[0]] += 1
        for i, count in enumerate(counts):
            self.presence[i] += count > 0

    def finish(self):
        if self.frames:
            yield self._record()
