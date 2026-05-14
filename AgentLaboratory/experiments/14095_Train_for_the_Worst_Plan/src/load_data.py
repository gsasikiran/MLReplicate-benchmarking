import datasets, random

# Load an external dataset (wikitext-2-raw-v1) from HuggingFace and select the first 20 examples.
data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(20))

# Maximum sequence length for the experiment.
max_len = 2048

# A simple preprocessing: converts text to string, tokenizes via whitespace, truncates or pads,
# and creates a "closer" permutation via L/10 random swaps.
def preprocess(example):
    # Ensure we get a string value from the 'text' field.
    text = str(example.get("text", ""))
    tokens = text.split()
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += ["<PAD>"] * (max_len - len(tokens))
    # Create an identity permutation.
    identity = list(range(max_len))
    # Create a "closer" permutation by performing max_len//10 random swaps.
    pi_closer = identity.copy()
    for _ in range(max_len // 10):
        i, j = random.sample(range(max_len), 2)
        pi_closer[i], pi_closer[j] = pi_closer[j], pi_closer[i]
    # Save the processed text and the permutation into the example.
    example["processed_text"] = " ".join(tokens)
    example["pi_closer"] = pi_closer
    return example

# Apply preprocessing to each example in the dataset.
processed_dataset = data.map(preprocess)

# Output the first processed example for verification.
print(processed_dataset[0])