"""Diagnostic comparison with an all-frame reference (object F1 is micro-averaged)."""
import argparse
import gzip
import json
from pathlib import Path


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    records, errors = {}, {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "error" in row:
                errors[row["video"]] = row["error"]
            else:
                key = row["video"], row["second"]
                if key in records:
                    raise ValueError(f"duplicate second: {key}")
                records[key] = row
    return records, errors


def compare(reference_path, candidate_path):
    reference, reference_errors = load(reference_path)
    candidate, candidate_errors = load(candidate_path)
    tp = fp = fn = exact = within_one = 0
    for key in reference.keys() | candidate.keys():
        expected, actual = reference.get(key), candidate.get(key)
        truth = set(expected["objects"]) if expected else set()
        predicted = set(actual["objects"]) if actual else set()
        tp += len(truth & predicted)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        if expected and actual:
            difference = abs(expected["people"] - actual["people"])
            exact += difference == 0
            within_one += difference <= 1
    total = len(reference)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    result = {"object_micro_f1": f1, "people_exact": exact / total if total else None,
              "people_within_one": within_one / total if total else None,
              "reference_seconds": total, "candidate_seconds": len(candidate),
              "missing_seconds": len(reference.keys() - candidate.keys()),
              "extra_seconds": len(candidate.keys() - reference.keys()),
              "reference_errors": len(reference_errors), "candidate_errors": len(candidate_errors)}
    result["passes_diagnostic"] = bool(total and f1 >= .95 and exact / total >= .85
        and within_one / total >= .95 and not result["missing_seconds"] and not result["extra_seconds"]
        and reference_errors.keys() == candidate_errors.keys())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passes_diagnostic"] else 1)
