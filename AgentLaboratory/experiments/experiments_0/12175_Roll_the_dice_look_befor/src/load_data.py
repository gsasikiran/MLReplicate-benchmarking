from datasets import load_dataset, Dataset
import random
import string

# Load an external dataset from HuggingFace (used here as a seed/reference resource)
external_dataset = load_dataset("ag_news", split="train")

# Helper: randomly choose a seed prefix based on provided conditions
seed_lengths = [0, 2, 4, 10]
prefix_options = ['seed', 'null', 'pause']
# Inline implementation without defining an explicit function
def get_seed_prefix():
    choice = random.choice(prefix_options)
    if choice == 'seed':
        L = random.choice(seed_lengths)
        seed_str = ''.join(random.choices(string.ascii_uppercase, k=L))
        return f"SEED: {seed_str}"
    elif choice == 'pause':
        return " ".join(["[PAUSE]"] * 10)
    else:
        return ""

# Sibling Discovery Data:
# Create a bipartite structure with 5 parents and a pool of lower-case child tokens.
parents = ["A", "B", "C", "D", "E"]
children_pool = list(string.ascii_lowercase)
sibling_samples = []
# Generate a small sample (10 examples for simplicity instead of 50k)
for _ in range(10):
    parent = random.choice(parents)
    # Select two distinct children from the pool
    child1, child2 = random.sample(children_pool, 2)
    # Randomly output either sibling-first order or reverse order (for ablation)
    if random.random() < 0.5:
        sample = f"({child1}, {child2}, {parent})"
    else:
        sample = f"({parent}, {child1}, {child2})"
    prefix = get_seed_prefix()
    sibling_samples.append((prefix + " " + sample).strip())

# Triangle Discovery Data:
# Build triangles over a small vertex set for simplicity (vertices 0-9)
triangle_samples = []
vertices = list(range(10))
for _ in range(10):
    tri_vertices = random.sample(vertices, 3)
    tri_vertices.sort()
    sample = f"tri: ({tri_vertices[0]},{tri_vertices[1]}),({tri_vertices[1]},{tri_vertices[2]}),({tri_vertices[2]},{tri_vertices[0]})"
    prefix = get_seed_prefix()
    triangle_samples.append((prefix + " " + sample).strip())

# Circle Construction Data:
# Create cycles using 9 edges from a vocabulary of 15 distinct tokens (first 15 lowercase letters)
vocab = list(string.ascii_lowercase[:15])
circle_samples = []
for _ in range(10):
    cycle = random.sample(vocab, 9)
    edges = []
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.append(f"{u}→{v}")
    sample = ", ".join(edges)
    prefix = get_seed_prefix()
    circle_samples.append((prefix + " " + sample).strip())

# Line Construction Data:
# Create a path (line) with 9 edges using 10 vertices sampled from the same vocabulary.
line_samples = []
for _ in range(10):
    path = random.sample(vocab, 10)
    edges = []
    for i in range(len(path) - 1):
        edges.append(f"{path[i]}→{path[i+1]}")
    sample = ", ".join(edges)
    prefix = get_seed_prefix()
    line_samples.append((prefix + " " + sample).strip())

# Combine generated samples into a dictionary
data_dict = {
    "sibling": sibling_samples,
    "triangle": triangle_samples,
    "circle": circle_samples,
    "line": line_samples,
}

# Create a HuggingFace Dataset from the dictionary
final_dataset = Dataset.from_dict(data_dict)

print("External dataset (ag_news) loaded with", len(external_dataset), "examples")
print("Synthetic dataset counts:", {k: len(v) for k, v in data_dict.items()})
print("Sample Sibling Discovery example:", sibling_samples[0])