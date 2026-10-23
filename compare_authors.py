import os

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_gutenberg
from download_data import BOOKS, DATA_DIR
from ngram_model import NGramModel

N = 4
K = 0.01


def book_path(author: str, title: str) -> str:
    return os.path.join(DATA_DIR, author, f"{title}.txt")


authors = list(BOOKS)
models = {}
test_texts = {}

# The first books of each author are for training, the last one is held out for testing
for author, books in BOOKS.items():
    *train_titles, test_title = books

    train = " ".join(load_gutenberg(book_path(author, t)) for t in train_titles)

    models[author] = NGramModel(n=N, k=K).fit(train)
    test_texts[author] = load_gutenberg(book_path(author, test_title))
    print(f"Trained {author} model, held out {test_title}")

results = np.zeros((len(authors), len(authors)))

for i, model_author in enumerate(authors):
    for j, text_author in enumerate(authors):
        results[i, j] = models[model_author].perplexity(test_texts[text_author])

relative = results / results.min(axis=0)

correct = sum(results[:, j].argmin() == j for j in range(len(authors)))
print(f"Correct author has the lowest perplexity for {correct}/{len(authors)} held-out books")

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(relative, cmap="magma")

for i in range(len(authors)):
    for j in range(len(authors)):
        is_best = results[i, j] == results[:, j].min()
        ax.text(j, i, f"{results[i, j]:.2f}", ha="center", va="center",
                color="black" if relative[i, j] > relative.mean() else "white",
                fontweight="bold" if is_best else "normal")

labels = [a.replace("_", " ").title() for a in authors]
ax.set_xticks(range(len(authors)), labels, rotation=30, ha="right")
ax.set_yticks(range(len(authors)), labels)
ax.set_xlabel("Held-out book's author")
ax.set_ylabel("Model trained on")
ax.set_title(f"Perplexity of each author's model on unseen books (n={N}, k={K})\n"
             f"Bold = best model for that book; {correct}/{len(authors)} correct")

fig.colorbar(im, ax=ax, label="Perplexity relative to best model for that book")
plt.savefig("author_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
