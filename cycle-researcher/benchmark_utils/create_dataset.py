import os
import json
import re
from PyPDF2 import PdfReader

# Paths
input_folder = "/nfs/home/<Username>/research/code/MLReplicate/dataset"        # Folder with PDFs
output_folder = "/nfs/home/<Username>/research/code/MLReplicate/cycle-researcher/dataset/"      # Folder to save JSONs

os.makedirs(output_folder, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_title(text):
    # Assumes the title is in the first few lines
    lines = text.split("\n")
    for line in lines[:10]:  # check first 10 lines
        clean_line = line.strip()
        if clean_line and len(clean_line.split()) > 3:  # skip very short lines
            return clean_line
    return "Unknown Title"

def extract_references(text):
    # Very simple heuristic: looks for "References" section
    refs = []
    match = re.search(r'(References|REFERENCES)(.*)', text, re.DOTALL)
    if match:
        ref_text = match.group(2)
        # Split by newlines and remove empty lines
        lines = [line.strip() for line in ref_text.split("\n") if line.strip()]
        refs.extend(lines)
    return refs

# Process PDFs
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        text = extract_text_from_pdf(pdf_path)
        title = extract_title(text)
        references = extract_references(text)
        
        data = {
            "title": title,
            "references": references
        }
        
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Processed {filename}: {title}, {len(references)} references")
