import random
from collections import Counter, defaultdict


class NGramModel:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("N-gram length must be positive")

        self.n = n
        self.counts = defaultdict(Counter)
        self.probs = {}

    def fit(self, corpus: str):
        self.counts = defaultdict(Counter)

        for i in range(len(corpus) - self.n):
            ngram = corpus[i: i+self.n]
            next_char = corpus[i+self.n]
            self.counts[ngram][next_char] += 1

        self.probs = {}

        for ngram, next_chars in self.counts.items():
            total = sum(next_chars.values())

            self.probs[ngram] = {
                char: count / total
                for char, count in next_chars.items()
            }

        return self

    def generate(self, length: int):
        if not self.probs:
            raise ValueError("Train the model first")

        if length < self.n:
            raise ValueError("Length must be >= n")

        text = random.choice(list(self.probs.keys()))

        for _ in range(length - self.n):
            ngram = text[-self.n:]

            if ngram not in self.probs:
                break

            next_chars = self.probs[ngram]

            next_char = random.choices(
                list(next_chars.keys()),
                weights=list(next_chars.values()),
                k=1
            )[0]

            text += next_char

        return text

if __name__ == "__main__":
    with open("data/LesMis.txt", encoding="utf-8") as f:
        corpus = f.read().replace("\n", " ")

    bigram_model = NGramModel(2)
    bigram_model.fit(corpus)
    print(bigram_model.generate(100))


    