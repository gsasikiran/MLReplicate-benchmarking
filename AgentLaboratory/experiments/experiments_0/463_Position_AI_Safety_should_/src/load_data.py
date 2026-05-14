from datasets import load_dataset

# Load an external HuggingFace dataset (using the 'imdb' dataset for illustration)
dataset = load_dataset("imdb", split="train[:100]")  # using a small subset for simplicity

# A simple mapping: assign each record a category based on text length,
# simulating a mapping to "automation-prone" vs "manual-intensive" as per our transparent taxonomy.
labeled_dataset = dataset.map(lambda example: {"category": "automation-prone" if len(example["text"]) > 150 else "manual-intensive"})

# Construct an 'event-like' dataset structure:
# Here we simulate an event dataset by creating dummy monthly count entries, 
# e.g., setting a fake 'month' field and a random indicator 'count' for demonstration.
def add_event_fields(example):
    # For simplicity, assigning each example a dummy month and count
    # In a real scenario, these would be derived from actual temporal and control variables.
    example["month"] = "2022-01"
    example["count"] = len(example["text"]) % 10  # dummy count based on text length
    return example

event_dataset = labeled_dataset.map(add_event_fields)

# Print one sample to verify
print(event_dataset[0])
print("Data preparation complete: External dataset loaded, categorized, and event fields added.")