import os
import argparse
import pypdf
import openai

SYSTEM_PROMPT = """You are an expert in scientific research and AI experiment design. 
                   Based on the following research paper text, generate a detailed 'task-notes' section for a YAML configuration file. The notes should guide an AI agent to reproduce or build upon the paper's experiments. The output should be in YAML format, with sections for 'plan-formulation', 'data-preparation', 'running-experiments', 'results-interpretation', and 'report-writing'.
                   
                   Take the following structure as reference:

                   research-topic: "Your goal is to design reasoning and prompt engineering techniques to maximize accuracy for a simple mnist classification task.

                   task-notes:
                        plan-formulation:
                            - 'Design a single experiment that enables gpt-4o-mini to classify MNIST digits from pixel data using novel prompting techniques.'
                            - 'Assume access to MNIST images but only as text-representable arrays (e.g., ASCII or flattened pixel arrays).'
                            - 'Your goal is to maximize classification accuracy through clever prompt design, even though gpt-4o-mini is not trained for vision.'
                            - 'Do NOT use any external vision models or CNNs.'
                            - 'Explore prompt structures such as analogical reasoning, verbal pattern matching, or step-by-step logical deduction from patterns in the pixel grid.'
                            - 'Use few-shot prompting with exemplars representing pixel-to-digit mappings.'
                            - 'Keep system prompt consistent as: "You are a highly intelligent classifier that can read and interpret visual patterns from text-based image inputs."'
                            - "DO NOT PLAN FOR TOO LONG. Submit your plan soon."

                        data-preparation:
                            - 'Convert 28x28 MNIST grayscale images to simple ASCII or binary grid format (e.g., "⬛⬜⬛⬛⬜...").'
                            - 'Or flatten image into a 784-length vector, normalize to binary or 0-9 grayscale tokens.'
                            - 'For few-shot prompts, include 3–5 examples formatted consistently as:\nInput Image:\n<ASCII grid>\nDigit: <label>'
                            - 'Create a batch of 500 test samples using `datasets.load_dataset("mnist", split="test")`.'
                            - 'Ensure class balance in test samples.'
                            - 'Keep pixel compression formats consistent across all prompts.'

                        running-experiments:
                            - "Use only gpt-4o-mini to perform inference with ASCII/binary formatted image prompts."
                            - "Use triple quotes ''' for all prompt strings."
                            - 'Query gpt-4o-mini with:\n```response = query_gpt4omini(prompt=prompt, system=system_prompt)```'
                            - "Prompt format should include:\n1. Few-shot examples\n2. The current image as ASCII or flattened array\n3. A final question: 'What digit does this image represent? Answer with a single number.'"
                            - 'Test with and without chain-of-thought reasoning like: "First I will count the number of filled pixels in each row... Then compare to known patterns."'
                            - 'Run prompt variants such as:\n  - Grid-to-digit analogy: "This pattern looks similar to 3 because of its top and bottom rows..."\n  - Verbal rules: "If top row is mostly filled and middle is empty, then likely a 7..."'
                            - "Use concurrent.futures for parallelizing 500 inference calls to gpt-4o-mini."
                            - "Log predictions and compare against true labels using exact match."

                        results-interpretation:
                            - 'Report final classification accuracy across the 500 MNIST test samples.'
                            - 'Analyze per-digit performance. Where does gpt-4o-mini struggle (e.g., 3 vs 8)?'
                            - 'Visualize attention to prompt parts using token-level saliency (if possible).'
                            - 'Compare few-shot prompt strategies: Which exemplars or grid formats helped the most?'
                            - 'Discuss reasoning patterns gpt-4o-mini might be implicitly using from prompts.'

                        report-writing:
                            - 'Report the final accuracy, confusion matrix, and few-shot prompt templates used.'
                            - 'Include example prompts that succeeded and failed (with LLM’s reasoning steps).'
                            - 'Plot performance vs number of few-shot examples.'
                            - 'Discuss feasibility of using LLMs for "textified" vision tasks and implications for prompt-based perception.'
                            - 'Make the figures vivid: use color-coded ASCII digit renderings and grid similarity maps.' """

def get_openai_api_key():
    """Gets the OpenAI API key from the environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return api_key

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def generate_task_notes_with_openai(pdf_text):
    """Generates task notes using OpenAI API."""
    if not pdf_text:
        return "# Could not generate task notes because PDF text could not be extracted."

    try:
        client = openai.OpenAI(api_key=get_openai_api_key())
        prompt = f"""Here is the paper text: {pdf_text}

Provide the output in the following YAML format:

research-topic: "Your goal is to design reasoning and prompt engineering techniques to maximize accuracy for <RESEARCH-MOTIVATION>."
task-notes:
  plan-formulation:
    - '...'
  data-preparation:
    - '...'
  running-experiments:
    - '...'
  results-interpretation:
    - '...'
  report-writing:
    - '...'    
"""

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"# Error generating task notes with OpenAI: {e}"

def create_yaml_config(pdf_path, output_dir):
    """
    Creates a YAML configuration file for a given PDF.

    Args:
        pdf_path (str): The full path to the input PDF file.
        output_dir (str): The directory where the YAML file will be saved.
    """
    base_name = os.path.basename(pdf_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    print(f"Processing {base_name}...")

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Generate task notes with OpenAI
    print("Generating task notes with OpenAI...")
    task_notes = generate_task_notes_with_openai(pdf_text)

    yaml_content = f"""
# If you want to have user input or be a human-in-the-loop
copilot-mode: True

# Here you can put your OpenAI API key--if you don't have one or OpenAI doesn't work for you, you can also instead use `deepseek-api-key`
api-key: "YOUR_OPENAI_API_KEY_HERE"
# or deepseek-api-key: "DEEPSEEK-API-KEY-HERE"
# Agent Laboratory backend
llm-backend: "o3-mini"
# Literature review backend
lit-review-backend: "o3-mini"

# Base language
language: "English"

# Number of arxiv papers to lit review
num-papers-lit-review: 5
# Total number of papers to write in sequence
num-papers-to-write: 1
# Do you want to run multiple agent labs in parallel?
parallel-labs: False

# Total mle-solver steps per lab
mlesolver-max-steps: 3
# Total paper-solver steps per lab
papersolver-max-steps: 1
# The lab index for this lab (used for parallel runs)
lab-index: 1
# If you want to load an existing save
load-existing: False
# If fail, run exception?
except-if-fail: False
# Compile latex into PDFs during paper-solver
compile-latex: False

{task_notes}
"""

    output_yaml_path = os.path.join(output_dir, f"{file_name_without_ext}.yaml")
    
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YAML config: {output_yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Create YAML configuration files for PDFs using OpenAI.")
    parser.add_argument("--pdf_dir", type=str, default="dataset", help="Directory containing PDF files.")
    parser.add_argument("--output_dir", type=str, default="AgentLaboratory/experiment_configs", help="Directory to save the YAML files.")
    
    args = parser.parse_args()

    try:
        get_openai_api_key() # Check for API key at the beginning
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not os.path.isdir(args.pdf_dir):
        print(f"Error: PDF directory not found at '{args.pdf_dir}'")
        return

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: '{args.output_dir}'")

    for filename in os.listdir(args.pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(args.pdf_dir, filename)
            create_yaml_config(pdf_path, args.output_dir)

if __name__ == "__main__":
    main()