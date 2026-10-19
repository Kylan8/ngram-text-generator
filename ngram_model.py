import random
from collections import Counter, defaultdict
import math


class NGramModel:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError("N-gram length must be positive")

        self.n = n
        self.counts = defaultdict(Counter)
        self.context_totals = Counter()
        self.vocab = set()

    def fit(self, corpus: str):
        
        
        self.counts = defaultdict(Counter)
        self.context_totals = Counter
        self.vocab = set(corpus)

        for i in range(len(corpus) - self.n):
            ngram = corpus[i: i+self.n]
            next_char = corpus[i+self.n]
            self.counts[ngram][next_char] += 1
            self.context_totals[ngram] += 1


        return self

    def prob(self, context: str, char: str) -> float:
        if context not in self.counts:
            return 0.0

        return self.counts[context][char] / self.context_totals[context]

    def generate(self, length: int):
        if not self.probs:
            raise ValueError("Train the model first")

        if length < self.n:
            raise ValueError("Length must be >= n")

        text = random.choice(list(self.probs.keys()))

        for _ in range(length - self.n):
            ngram = text[-self.n:]

            if ngram not in self.counts:
                break

            next_chars = self.counts[ngram]

            next_char = random.choices(
                list(next_chars.keys()),
                weights=list(next_chars.values()),
                k=1
            )[0]

            text += next_char

        return text

    def perplexity(self, text):

        total = 0
        count = 0

        for i in range(len(corpus) - self.n):
            ngram = corpus[i: i+self.n]
            next_char = corpus[i+self.n]

            total += math.log(self.prob(ngram, next_char))
            count += 1

        return math.exp(-total / count)


if __name__ == "__main__":
    with open("data/LesMis.txt", encoding="utf-8") as f:
        corpus = f.read().replace("\n", " ")

    bigram_model = NGramModel(2)
    bigram_model.fit(corpus)
    print(bigram_model.generate(100))


    