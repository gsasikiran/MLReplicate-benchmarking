from datasets import load_dataset, Dataset
import random
import string

# Load an external dataset (using a small portion of CNN/DailyMail for demonstration)
external_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")

# Function to randomly choose a seed prefix
def get_seed_prefix():
    choice = random.choice(["none", "seed", "pause"])
    if choice == "seed":
        length = random.choice([2, 4, 10])
        return "SEED: " + "".join(random.choices(string.ascii_uppercase, k=length))
    elif choice == "pause":
        return "PAUSE: " + " ".join(["[PAUSE]"] * 10)
    else:
        return ""

# Sibling Discovery Data: Create a small bipartite graph example.
# For simplicity, use 5 parents (A-E) and 20 children per parent, producing limited examples.
parents = ["A", "B", "C", "D", "E"]
sibling_data = []
for parent in parents:
    children = [str(i) for i in range(1, 21)]
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            # Format: "(γ, γ′, Γ)" in sibling-first order
            example_str = f"{children[i]} {children[j]} {parent}"
            prefix = get_seed_prefix()
            full_example = (prefix + " " if prefix else "") + example_str
            sibling_data.append({"task": "sibling", "example": full_example})
            if len(sibling_data) >= 50:  # Limit to 50 examples for simplicity.
                break
        if len(sibling_data) >= 50:
            break
    if len(sibling_data) >= 50:
        break

# Triangle Discovery Data: Create triangles in an undirected graph.
triangle_data = []
for _ in range(50):
    vertices = sorted(random.sample(range(1, 50), 3))
    # Represent a triangle with the given format.
    triangle_str = f"tri: ({vertices[0]},{vertices[1]}),({vertices[1]},{vertices[2]}),({vertices[2]},{vertices[0]})"
    prefix = get_seed_prefix()
    full_example = (prefix + " " if prefix else "") + triangle_str
    triangle_data.append({"task": "triangle", "example": full_example})

# Circle Construction Data: Create samples of a cycle with 9 edges.
circle_data = []
for _ in range(50):
    vertices = random.sample(range(1, 20), 9)
    edges = []
    for i in range(9):
        u = vertices[i]
        v = vertices[(i + 1) % 9]
        edges.append(f"{u}→{v}")
    circle_str = ", ".join(edges)
    prefix = get_seed_prefix()
    full_example = (prefix + " " if prefix else "") + circle_str
    circle_data.append({"task": "circle", "example": full_example})

# Line Construction Data: Create samples of a line (path) with 8 edges (9 vertices).
line_data = []
for _ in range(50):
    vertices = random.sample(range(1, 20), 9)
    edges = []
    for i in range(8):
        u = vertices[i]
        v = vertices[i + 1]
        edges.append(f"{u}→{v}")
    line_str = ", ".join(edges)
    prefix = get_seed_prefix()
    full_example = (prefix + " " if prefix else "") + line_str
    line_data.append({"task": "line", "example": full_example})

# Combine all synthetic data into one dataset.
synthetic_examples = sibling_data + triangle_data + circle_data + line_data
synthetic_dataset = Dataset.from_dict({
    "task": [entry["task"] for entry in synthetic_examples],
    "example": [entry["example"] for entry in synthetic_examples]
})

print("External dataset sample:", external_dataset[0])
print("Synthetic dataset size:", len(synthetic_examples))