import argparse
import sys

from .counter import count_file


def build_parser():
    p = argparse.ArgumentParser(
        prog="wcx", description="Count lines, words and characters."
    )
    p.add_argument("files", nargs="+", help="files to count")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    total = [0, 0, 0]
    for path in args.files:
        try:
            counts = count_file(path)
        except OSError as e:
            print(f"wcx: {path}: {e.strerror}", file=sys.stderr)
            return 1
        for i, c in enumerate(counts):
            total[i] += c
        print(f"{counts[0]}\t{counts[1]}\t{counts[2]}\t{path}")
    if len(args.files) > 1:
        print(f"{total[0]}\t{total[1]}\t{total[2]}\ttotal")
    return 0
