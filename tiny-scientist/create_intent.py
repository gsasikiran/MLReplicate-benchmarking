#!/usr/bin/env python3
"""
Abstract to Intent Converter
Converts research paper abstracts into tiny-scientist friendly intents
"""

import os
import sys
from openai import OpenAI

def convert_abstract_to_intent(abstract: str, api_key: str) -> str:
    """
    Convert a paper abstract to a research intent using GPT-4o
    
    Args:
        abstract: The paper abstract text
        api_key: OpenAI API key
        
    Returns:
        The converted intent string
    """
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""I have a research paper abstract. Convert it into a brief "research intent" statement (1-2 sentences) that describes:
1. The main research goal
2. The main research questioon

Paper Abstract:
{abstract}

Output ONLY the intent statement, nothing else. No explanation, no labels, just the intent."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts research abstracts into concise, implementation-focused research intents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        intent = response.choices[0].message.content.strip()
        return intent
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        sys.exit(1)


def main():
    """Main function"""
    
    print("="*60)
    print("Abstract to Intent Converter")
    print("="*60)
    print()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        print()
        print("Set it with:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print()
        print("Or pass it as argument:")
        print("  python abstract_to_intent.py --api-key your-key")
        sys.exit(1)
    
    # Check for command line API key
    if len(sys.argv) > 2 and sys.argv[1] == "--api-key":
        api_key = sys.argv[2]
    
    print("Enter your paper abstract (press Ctrl+D when done):")
    print("-"*60)
    
    # Read abstract from stdin
    try:
        abstract = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    
    if not abstract:
        print("Error: No abstract provided!")
        sys.exit(1)
    
    print()
    print("Converting abstract to intent...")
    print()
    
    # Convert
    intent = convert_abstract_to_intent(abstract, api_key)
    
    print("="*60)
    print("INTENT:")
    print("="*60)
    print(intent)
    print("="*60)
    print()
    print("Copy this intent to your intents.txt file!")
    print()


def batch_convert(input_file: str, output_file: str, api_key: str):
    """
    Batch convert multiple abstracts from a file
    
    Args:
        input_file: File with abstracts (separated by blank lines)
        output_file: Output file for intents
        api_key: OpenAI API key
    """
    
    print(f"Reading abstracts from: {input_file}")
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split by double newlines (blank lines)
    abstracts = [a.strip() for a in content.split('\n\n') if a.strip()]
    
    print(f"Found {len(abstracts)} abstracts")
    print()
    
    intents = []
    
    for i, abstract in enumerate(abstracts, 1):
        print(f"Converting abstract {i}/{len(abstracts)}...")
        
        # Show first 100 chars
        preview = abstract[:100] + "..." if len(abstract) > 100 else abstract
        print(f"  Preview: {preview}")
        
        intent = convert_abstract_to_intent(abstract, api_key)
        intents.append(intent)
        
        print(f"  Intent: {intent}")
        print()
    
    # Save to file
    with open(output_file, 'w') as f:
        for intent in intents:
            f.write(intent + '\n')
    
    print("="*60)
    print(f"Saved {len(intents)} intents to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    
    # Check if batch mode
    if len(sys.argv) >= 3 and sys.argv[1] == "--batch":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set!")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "intents.txt"
        
        batch_convert(input_file, output_file, api_key)
    else:
        main()