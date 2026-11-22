"""Download selected FineVideo shards after the user has accepted dataset terms."""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("finevideo"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--revision", default="main", help="Dataset commit to reproduce a download")
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 50:
        parser.error("select a nonempty range within the first 50 shards")
    for i in range(args.start, args.start + args.count):
        path = hf_hub_download("HuggingFaceFV/finevideo", repo_type="dataset",
                               filename=f"data/train-{i:05d}-of-01357.parquet",
                               revision=args.revision, local_dir=args.output)
        print(path, flush=True)


if __name__ == "__main__":
    main()
