import matplotlib.pyplot as plt

from ngram_model import NGramModel

k_values = [10**i for i in range(-3, 1)]
n_values = range(1, 7)

TRAIN_TEST_SPLIT = 0.9

with open("data/LesMis.txt", encoding="utf-8") as f:
    corpus = f.read().replace("\n", " ")

split = int(TRAIN_TEST_SPLIT*len(corpus))

train = corpus[:split]
test = corpus[split:]

results = {k: [] for k in k_values}

for n in n_values:
    for k_value in k_values:
        curr_model = NGramModel(n=n, k=k_value).fit(train)
        pp = curr_model.perplexity(test)
        results[k_value].append(pp)

best_pp = float("inf")
for k_value, pps in results.items():
    for n, pp in zip(n_values, pps):
        if pp < best_pp:
            best_pp, best_n, best_k = pp, n, k_value


for k_value, pps in results.items():
    plt.plot(n_values, pps, marker="o", label=f"k={k_value}")

plt.scatter([best_n], [best_pp], color="black", marker="*", s=250, zorder=3,
            label=f"Best: n={best_n}, k={best_k}")
plt.annotate(f"PP={best_pp:.2f}", xy=(best_n, best_pp),
             xytext=(10, 10), textcoords="offset points")

plt.xlabel("n (context length)")
plt.ylabel("Test perplexity")
plt.title("Test perplexity vs. n for different k")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("perplexity_vs_n_and_k.png", dpi=150, bbox_inches="tight")
plt.show()