import argparse
import sys

from .counter import count_file, read_text, top_words


def positive_int(s):
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def build_parser():
    p = argparse.ArgumentParser(prog="wcx", description="Count lines, words and characters.")
    p.add_argument("files", nargs="+", help="files to count")
    p.add_argument("--top", type=positive_int, metavar="N", help="print the N most frequent words")
    p.add_argument("-i", "--ignore-case", action="store_true", help="count words case-insensitively")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    total = [0, 0, 0]
    texts = []
    for path in args.files:
        try:
            counts = count_file(path)
            if args.top:
                texts.append(read_text(path))
        except OSError as e:
            print(f"wcx: {path}: {e.strerror}", file=sys.stderr)
            return 1
        for i, c in enumerate(counts):
            total[i] += c
        print(f"{counts[0]}\t{counts[1]}\t{counts[2]}\t{path}")
    if len(args.files) > 1:
        print(f"{total[0]}\t{total[1]}\t{total[2]}\ttotal")
    if args.top:
        for word, n in top_words(texts, args.top, args.ignore_case):
            print(f"{n}\t{word}")
    return 0
