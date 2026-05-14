import os
import sys
from pathlib import Path
from datetime import datetime
import json
import PyPDF2
from openai import OpenAI

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            num_pages = len(pdf_reader.pages)
            print(f"Reading all {num_pages} pages from the PDF...")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def calculate_cost(input_tokens, output_tokens, model="gpt-4o"):
    """
    Calculate the cost of the API call based on token usage.
    
    GPT-4o pricing (as of 2024):
    - Input: $2.50 per 1M tokens
    - Output: $10.00 per 1M tokens
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name (for future extensibility)
        
    Returns:
        Dictionary with cost breakdown
    """
    input_price_per_1m = 2.50
    output_price_per_1m = 10.00
    
    input_cost = (input_tokens / 1_000_000) * input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * output_price_per_1m
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def save_cost_log(pdf_path, title, cost_info, log_file="cost_log.json"):
    """
    Save cost information to a JSON log file.
    
    Args:
        pdf_path: Path to the PDF file
        title: Generated title
        cost_info: Cost information dictionary
        log_file: Path to the log file
    """
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": os.path.basename(pdf_path),
            "generated_title": title,
            "input_tokens": cost_info['input_tokens'],
            "output_tokens": cost_info['output_tokens'],
            "total_tokens": cost_info['total_tokens'],
            "total_cost_usd": round(cost_info['total_cost'], 6)
        }
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"\n✓ Cost logged to: {log_file}")
        
    except Exception as e:
        print(f"\nWarning: Could not save cost log: {e}")


def generate_title_with_gpt4o(paper_content, api_key):
    """
    Generate a title for the research paper using GPT-4o.
    
    Args:
        paper_content: Extracted text from the research paper
        api_key: OpenAI API key
        
    Returns:
        Tuple of (generated title, cost_info dictionary)
    """
    try:
        client = OpenAI(api_key=api_key)
        
        content_to_use = paper_content[:15000]
        prompt = f"""
You are an expert academic editor known for crafting clear, memorable, and field-appropriate research paper titles.

Below are examples of strong, concise titles:

1. "Attention Is All You Need"
2. "ImageNet Classification with Deep Convolutional Neural Networks"
3. "Large Language Models Are Zero-Shot Reasoners"

Using a similar style and tone, generate a title for the research paper below that:
- Expresses the paper’s main contribution or finding in a direct, impactful way
- Reflects the focus or approach without unnecessary words
- Avoids phrases like “A Study of”, “An Analysis of”, or “Research on”
- Does not include quotation marks or the word “title”

Research Paper Content:
{content_to_use}

Generate only the title, nothing else.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert academic editor specializing in creating research paper titles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        title = response.choices[0].message.content.strip()
        
        title = title.strip('"').strip("'")
        
        usage = response.usage
        cost_info = calculate_cost(usage.prompt_tokens, usage.completion_tokens)
        
        return title, cost_info
        
    except Exception as e:
        print(f"Error generating title with GPT-4o: {e}")
        return None, None

def main():
    """Main function to run the title generator."""
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_title_generator.py <path_to_pdf>")
        print("Example: python pdf_title_generator.py research_paper.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it using: export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print(f"Processing PDF: {pdf_path}")
    print("-" * 60)
    
    print("Extracting text from PDF...")
    paper_content = extract_text_from_pdf(pdf_path)
    
    if not paper_content:
        print("Failed to extract content from PDF.")
        sys.exit(1)
    
    print(f"Extracted {len(paper_content)} characters from the PDF.")
    print("-" * 60)
    
    print("Generating title using GPT-4o...")
    title, cost_info = generate_title_with_gpt4o(paper_content, api_key)
    
    if title and cost_info:
        print("-" * 60)
        print("GENERATED TITLE:")
        print(f"\n{title}\n")
        print("-" * 60)
        print("\nAPI USAGE & COST:")
        print(f"  Input tokens:  {cost_info['input_tokens']:,}")
        print(f"  Output tokens: {cost_info['output_tokens']:,}")
        print(f"  Total tokens:  {cost_info['total_tokens']:,}")
        print(f"\n  Input cost:    ${cost_info['input_cost']:.6f}")
        print(f"  Output cost:   ${cost_info['output_cost']:.6f}")
        print(f"  Total cost:    ${cost_info['total_cost']:.6f}")
        print("-" * 60)
        
        save_cost_log(pdf_path, title, cost_info)
        
    else:
        print("Failed to generate title.")
        sys.exit(1)

if __name__ == "__main__":
    main()