import random
from collections import Counter, defaultdict
import math


class NGramModel:
    def __init__(self, n: int, k = 1):
        if n < 1:
            raise ValueError("N-gram length must be positive")

        self.k = k
        self.n = n
        self.counts = defaultdict(Counter)
        self.context_totals = Counter()
        self.vocab = set()

    def fit(self, corpus: str):
        
        
        self.counts = defaultdict(Counter)
        self.context_totals = Counter()
        self.vocab = set(corpus)

        for i in range(len(corpus) - self.n):
            ngram = corpus[i: i+self.n]
            next_char = corpus[i+self.n]
            self.counts[ngram][next_char] += 1
            self.context_totals[ngram] += 1


        return self

    def prob(self, context: str, char: str) -> float:
        return (self.counts[context][char] + self.k) / (self.context_totals[context] + self.k * len(self.vocab))

    def generate(self, length: int):
        if not self.counts:
            raise ValueError("Train the model first")

        if length < self.n:
            raise ValueError("Length must be >= n")

        text = random.choice(list(self.counts.keys()))

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

        for i in range(len(text) - self.n):
            ngram = text[i: i+self.n]
            next_char = text[i+self.n]

            if (self.prob(ngram, next_char) == 0):
                return float('inf')

            total += math.log(self.prob(ngram, next_char))
            count += 1

        return math.exp(-total / count)


if __name__ == "__main__":
    TRAIN_TEST_SPLIT = 0.9
   
    with open("data/LesMis.txt", encoding="utf-8") as f:
        corpus = f.read().replace("\n", " ")

    split = int(TRAIN_TEST_SPLIT*len(corpus))

    train = corpus[:split]
    test = corpus[split:]

    bigram_model = NGramModel(2)
    bigram_model.fit(train)
    print(bigram_model.perplexity(test))


    