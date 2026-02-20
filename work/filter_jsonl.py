#!/usr/bin/env python3

import argparse
import json
import re
from typing import List


def compile_pattern(
    words: List[str],
    case_sensitive: bool,
    substring: bool,
) -> re.Pattern:
    """
    Create a regex pattern that matches any of the given words.
    If substring=True → match anywhere
    If substring=False → match whole words only
    """
    escaped_words = [re.escape(word) for word in words]

    if substring:
        pattern_str = "(" + "|".join(escaped_words) + ")"
    else:
        pattern_str = r"\b(" + "|".join(escaped_words) + r")\b"

    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern_str, flags)


def line_contains_words(obj: dict, pattern: re.Pattern) -> bool:
    """
    Check if any value in the JSON object contains the filtered words.
    """
    text = json.dumps(obj, ensure_ascii=False)
    return bool(pattern.search(text))


def filter_jsonl(
    input_path: str,
    output_path: str,
    words: List[str],
    case_sensitive: bool,
    remove_matching: bool,
    substring: bool,
):
    pattern = compile_pattern(words, case_sensitive, substring)

    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            total += 1

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

            contains = line_contains_words(obj, pattern)

            # Logic:
            # remove_matching=True  → drop matching lines
            # remove_matching=False → keep only matching lines
            if (contains and not remove_matching) or (not contains and remove_matching):
                outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Processed: {total}")
    print(f"Written:   {kept}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Filter JSONL file by certain words.")

    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument(
        "--words",
        nargs="+",
        required=True,
        help="Words to filter on",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive matching",
    )
    parser.add_argument(
        "--remove-matching",
        action="store_true",
        help="Remove lines that contain the words (default keeps only matching lines)",
    )
    parser.add_argument(
        "--substring",
        action="store_true",
        help="Match substrings instead of whole words",
    )

    args = parser.parse_args()

    filter_jsonl(
        input_path=args.input,
        output_path=args.output,
        words=args.words,
        case_sensitive=args.case_sensitive,
        remove_matching=args.remove_matching,
        substring=args.substring,
    )


if __name__ == "__main__":
    main()
