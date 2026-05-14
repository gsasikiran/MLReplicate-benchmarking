from datasets import load_dataset
import json

# Prepare a smaller subset to reduce processing time.

# MediumDocEdit-Chat: Use 10 articles from the Medium dataset.
try:
    medium_data = load_dataset("fabiochiu/medium-articles", split="v0.1.0_hf")
    medium_subset = medium_data.select(range(10))
except Exception as error:
    print("Error loading Medium dataset:", error)
    medium_subset = []

doc_cards = []
for article in medium_subset:
    content = article.get("content", "") or article.get("article", "")
    doc_cards.append({
        "task_description": "Document editing: Summarize the article.",
        "target_problem": content[:80],  # use first 80 characters as a simple summary
        "evaluation_rubric": "BLEU score against a reference summary.",
        "constraints": "Max summary length: 200 words."
    })

# BigCodeBench-Chat: Use 20 problems from BigCodeBench or simulated dummy data.
try:
    code_data = load_dataset("bigcode/bigcodebench", split="v0.1.0_hf")
    code_subset = code_data.select(range(20))
except Exception as error:
    print("Error loading BigCodeBench dataset:", error)
    code_subset = [{"problem": "Implement a function to add two numbers."}] * 20

code_cards = []
for item in code_subset:
    problem_text = item.get("problem", "Solve the code challenge.")
    code_cards.append({
        "task_description": "Code generation: Produce runnable code.",
        "target_problem": problem_text,
        "evaluation_rubric": "Unit tests pass and static analysis check.",
        "constraints": "Rename function to 'task_func'."
    })

# MATH-Chat: Use 10 math problems from the MATH dataset or dummy data.
try:
    math_data = load_dataset("hendrycks/math", split="v0.1.0_hf")
    math_subset = math_data.select(range(10))
except Exception as error:
    print("Error loading MATH dataset:", error)
    math_subset = [{"problem": "2+2", "answer": "4"}] * 10

math_cards = []
for item in math_subset:
    problem_text = item.get("problem", "Solve the math problem.")
    math_cards.append({
        "task_description": "Math problem solving.",
        "target_problem": problem_text,
        "evaluation_rubric": "Exact match with ground truth.",
        "constraints": "Answer in numeric or standard symbolic format."
    })

# Abg-CoQA: Simulate 5 examples for ambiguity resolution.
abg_cards = []
for i in range(5):
    abg_cards.append({
        "task_description": "Dialogue ambiguity resolution.",
        "target_problem": f"Story {i} with an ambiguous question.",
        "evaluation_rubric": "Clarity and correctness in resolving ambiguity.",
        "constraints": "Expected label: non-ambiguous."
    })

# Conversation scaffolding components
user_simulator_prompt = {
  "current_answer": "",
  "thought": "Analyze the goal card and decide the next move.",
  "response": "",
  "terminal_signal": False
}
assistant_system_prompt = "You are a proactive assistant in a multiturn dialogue. Provide clear and interactive responses."
proactive_base_prompt = {
  "current_problem": "",
  "thought": "",
  "response": ""
}

# Combine all prepared data into one JSON structure.
prepared_data = {
  "MediumDocEdit-Chat": doc_cards,
  "BigCodeBench-Chat": code_cards,
  "MATH-Chat": math_cards,
  "Abg-CoQA": abg_cards,
  "user_simulator_prompt": user_simulator_prompt,
  "assistant_system_prompt": assistant_system_prompt,
  "proactive_base_prompt": proactive_base_prompt
}

print(json.dumps(prepared_data, indent=2))