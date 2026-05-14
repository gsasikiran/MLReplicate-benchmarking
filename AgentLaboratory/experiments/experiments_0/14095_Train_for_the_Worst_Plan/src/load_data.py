import random
import math
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# -----------------------------
# Prepare text dataset using an external HuggingFace dataset
# -----------------------------
# Load the Wikitext-2 dataset as a stand-in for SlimPajama (cleaned RedPajama)
text_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Load a tokenizer that uses learned positional embeddings (e.g., RoBERTa uses learned embeddings)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Set maximum sequence length. For scaling laws and π-learner experiments, we use L=2048.
max_length = 2048

# Tokenize the text using a simple mapping function
def tokenize_example(example):
    return tokenizer(example["text"], truncation=True, max_length=max_length)

tokenized_text = text_dataset.map(tokenize_example, batched=True)

# Define a helper function to sample a permutation (π) for different regimes.
def sample_permutation(sequence_length, regime):
    pi = list(range(sequence_length))
    if regime == "Unif":
        num_swaps = int(sequence_length * math.log(sequence_length + 1))
    elif regime == "Closer":
        num_swaps = sequence_length // 10
    elif regime == "Much-closer":
        num_swaps = int(math.sqrt(sequence_length))
    else:
        num_swaps = 0
    for _ in range(num_swaps):
        i, j = random.sample(range(sequence_length), 2)
        pi[i], pi[j] = pi[j], pi[i]
    return pi

# Attach sampled π permutations to each tokenized example to ensure determinism per batch.
def add_permutations(example):
    seq_len = len(example["input_ids"])
    example["pi_unif"] = sample_permutation(seq_len, "Unif")
    example["pi_closer"] = sample_permutation(seq_len, "Closer")
    example["pi_much_closer"] = sample_permutation(seq_len, "Much-closer")
    return example

tokenized_text = tokenized_text.map(add_permutations)

# -----------------------------
# Prepare synthetic L&O-NAE-SAT puzzles
# -----------------------------
# For synthetic puzzles, create sequences with two segments:
#   - N latent tokens (sampled over m classes)
#   - P observation tokens (here simulated as random integers to mimic NAE constraints)
# Then pad each sequence to a fixed length (512 tokens) to match model context length.

# Define multiple (N,P) settings.
np_settings = [(25, 275), (30, 270), (40, 260), (50, 250), (100, 200)]
m = 10  # number of latent classes
synthetic_examples = []

for (N, P) in np_settings:
    L = N + P
    # Generate latent tokens: values in range [0, m)
    latent_tokens = [random.randint(0, m - 1) for _ in range(N)]
    # Generate observation tokens: here we simulate with values in range [m, m+9]
    observation_tokens = [random.randint(m, m + 9) for _ in range(P)]
    tokens = latent_tokens + observation_tokens
    # Pad the sequence to length 512 with a constant token, e.g., 2
    if len(tokens) < 512:
        tokens = tokens + [2] * (512 - len(tokens))
    
    # For reproducibility, also attach a π permutation for one regime (e.g., "Unif")
    pi = sample_permutation(len(tokens), "Unif")
    
    synthetic_examples.append({
        "np_setting": f"N={N}_P={P}",
        "input_ids": tokens,
        "pi": pi
    })

synthetic_dataset = Dataset.from_dict({
    "np_setting": [ex["np_setting"] for ex in synthetic_examples],
    "input_ids": [ex["input_ids"] for ex in synthetic_examples],
    "pi": [ex["pi"] for ex in synthetic_examples]
})

# -----------------------------
# Print a sample from both datasets to verify preparation.
# -----------------------------
print("Text Dataset Sample:")
print(tokenized_text[0])
print("\nSynthetic Dataset Sample:")
print(synthetic_dataset[0])