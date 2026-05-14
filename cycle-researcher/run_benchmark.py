import os
import json
from benchmark_utils.main import main

def run_with_retry(topic, references, filename, max_retries=2):
    """
    Try running `main` up to `max_retries` times.
    Return number of failures.
    """
    failures = 0
    for attempt in range(max_retries):
        try:
            main(topic=topic, references=references, filename=filename)
            return failures  # success, stop retrying
        except Exception as e:
            failures += 1
            print(f"Error processing {filename} (attempt {attempt+1}): {e}")
    return failures  # failed all retries


if __name__ == "__main__":
    failure_log = {}
    print("Starting benchmark...")

    for filename in os.listdir("dataset"):
        if filename.endswith(".json"):
            filepath = os.path.join("dataset", filename)
            with open(filepath, "r") as f:
                try:
                    data_json = json.load(f)
                    title = data_json.get("title", None)
                    references_path = data_json.get("references", None)
                    with open(references_path, "r", encoding="utf-8") as ref_file:
                        references = ref_file.read()
                    print(f"Running benchmark for {filename} with title: {title}")

                    failures = run_with_retry(title, references, filename, max_retries=2)
                    if failures > 0:
                        failure_log[filename] = failures

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    failure_log[filename] = 2  # couldn't even parse JSON → full fail

    # Save failure log to JSON
    with open("experiments/failure_log.json", "w") as out:
        json.dump(failure_log, out, indent=2)

    print("Failure log saved to failure_log.json")