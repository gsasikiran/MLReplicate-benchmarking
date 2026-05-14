from datasets import load_dataset
from datetime import datetime, timedelta
import random

# Load an external HuggingFace dataset (using "ag_news" as a proxy for open job narrative data)
dataset = load_dataset("ag_news", split="train")

# Define the ChatGPT event date and a window of 4 weeks before and after
event_date = datetime(2022, 11, 30)
start_date = event_date - timedelta(weeks=4)
end_date = event_date + timedelta(weeks=4)

# Add a dummy "date" field to simulate timeline data; assign a random date within our extended window.
def add_dummy_date(example):
    random_date = start_date + (end_date - start_date) * random.random()
    example["date"] = random_date.strftime('%Y-%m-%d')
    return example

dataset = dataset.map(add_dummy_date)

# Filter the dataset for entries within the event window (4 weeks before and after the event)
def filter_event(example):
    example_date = datetime.strptime(example["date"], '%Y-%m-%d')
    return start_date <= example_date <= end_date

filtered_dataset = dataset.filter(filter_event)

print("Prepared dataset samples within the event window:", len(filtered_dataset))
print("Sample entry:", filtered_dataset[0])