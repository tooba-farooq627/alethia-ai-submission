"""Copy a few complete video rows into a small, separate validation input."""
import argparse
from pathlib import Path
import pyarrow.parquet as pq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("rows must be positive")
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    source = pq.ParquetFile(args.source)
    destination = args.output / args.source.name
    with pq.ParquetWriter(destination, source.schema_arrow) as writer:
        for i, batch in enumerate(source.iter_batches(batch_size=1)):
            if i >= args.rows:
                break
            writer.write_batch(batch)
    print(destination)
