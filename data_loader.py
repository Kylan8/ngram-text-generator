import re


def load_gutenberg(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    start = re.search(r"^\*\*\* ?START OF.*$", text, re.MULTILINE)
    end = re.search(r"^\*\*\* ?END OF.*$", text, re.MULTILINE)

    if start is None or end is None:
        raise ValueError(f"Could not find Gutenberg start/end markers in {path}")

    text = text[start.end():end.start()]

    return " ".join(text.split())


def split_into_passages(text: str, length: int) -> list[str]:
    if length < 1:
        raise ValueError("Passage length must be positive")

    return [
        text[i: i + length]
        for i in range(0, len(text) - length + 1, length)
    ]