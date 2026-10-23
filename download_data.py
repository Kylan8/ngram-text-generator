import os
import time
import urllib.request

BOOKS = {
    "austen": {
        "pride_and_prejudice": 1342,
        "emma": 158,
        "persuasion": 105,
    },
    "dickens": {
        "great_expectations": 1400,
        "oliver_twist": 730,
        "a_tale_of_two_cities": 98,
    },
    "twain": {
        "tom_sawyer": 74,
        "huckleberry_finn": 76,
        "connecticut_yankee": 86,
    },
    "doyle": {
        "adventures_of_sherlock_holmes": 1661,
        "hound_of_the_baskervilles": 2852,
        "a_study_in_scarlet": 244,
    },
    "charlotte_bronte": {
        "jane_eyre": 1260,
        "villette": 9182,
        "the_professor": 1028,
    },
    "melville": {
        "moby_dick": 2701,
        "typee": 1900,
        "omoo": 4045,
    },
}

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
DATA_DIR = "data"


def download_book(book_id: int, path: str):
    request = urllib.request.Request(
        URL.format(id=book_id),
        headers={"User-Agent": "ngram-text-generator (personal research project)"},
    )

    with urllib.request.urlopen(request) as response:
        text = response.read().decode("utf-8-sig")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def header_field(text: str, field: str) -> str:
    for line in text.splitlines()[:100]:
        if line.startswith(field + ":"):
            return line[len(field) + 1:].strip()

    return "?"


if __name__ == "__main__":
    for author, books in BOOKS.items():
        os.makedirs(os.path.join(DATA_DIR, author), exist_ok=True)

        for title, book_id in books.items():
            path = os.path.join(DATA_DIR, author, f"{title}.txt")

            if os.path.exists(path):
                print(f"Already have {path}, skipping")
                continue

            text = download_book(book_id, path)
            print(f"Saved {path}: {header_field(text, 'Title')} by {header_field(text, 'Author')}")

            time.sleep(2)