from paper_agent.methodology_composing_using_template import methodology_composing
from paper_agent.related_work_composing_using_template import related_work_composing
from paper_agent.experiments_composing import experiments_composing
from paper_agent.introduction_composing import introduction_composing
from paper_agent.conclusion_composing import conclusion_composing
from paper_agent.abstract_composing import abstract_composing
import asyncio
import argparse
import os
import shutil
from paper_agent.writing_fix import clean_tex_files_in_folder, process_tex_file
from paper_agent.tex_writer import compile_latex_project
from integration_helper import get_directory_manager

def create_main_latex_template(target_folder: str, research_field: str, instance_id: str):
    """Copy the main LaTeX template from templates directory."""
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # paper_agent/
    project_root = os.path.dirname(script_dir)  # AI-Researcher/
    template_source = os.path.join(project_root, "templates", "iclr2025", "iclr2025_conference_template.tex")
    dst = os.path.join(target_folder, "iclr2025_conference.tex")
    
    if os.path.exists(template_source):
        shutil.copy2(template_source, dst)
        print(f"Copied ICLR template: {dst}")
    else:
        raise FileNotFoundError(f"ICLR template not found at {template_source}. Please ensure template files are in templates/iclr2025/")

def copy_style_files(target_folder: str):
    """Copy ICLR style files from templates directory to target folder."""
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # paper_agent/
    project_root = os.path.dirname(script_dir)  # AI-Researcher/
    style_source = os.path.join(project_root, "templates", "iclr2025")
    
    if not os.path.exists(style_source):
        raise FileNotFoundError(f"Template directory not found: {style_source}")
    
    for style_file in os.listdir(style_source):
        if not style_file.startswith('.'): 
            src = os.path.join(style_source, style_file)
            dst = os.path.join(target_folder, style_file)
            shutil.copy2(src, dst)
            print(f"Copied template file: {style_file}")

def create_bibliography_file(target_folder: str):
    """Create bibliography file from research references."""
    # The bibliography should come from actual research references
    # For now, create empty file that will be populated by the composing functions
    bib_file_path = os.path.join(target_folder, "iclr2025_conference.bib")
    
    if not os.path.exists(bib_file_path):
        with open(bib_file_path, 'w', encoding='utf-8') as f:
            f.write("% Bibliography entries will be added by section composers\n")
        print(f"Created bibliography file: {bib_file_path}")

async def writing(research_field: str, instance_id: str):
    """Main paper writing function."""
    # Get centralized directories
    manager = get_directory_manager()
    
    # Ensure run directory exists
    if not manager.current_run_dir:
        # For standalone paper generation, create a paper-only run directory
        import logging
        logging.warning(f"No active run directory found. Creating standalone paper generation directory.")
        manager.create_run_directory(f"paper_{research_field}_{instance_id}")
    
    target_folder = str(manager.get_paper_dir() / f"{research_field}_{instance_id}")
    os.makedirs(target_folder, exist_ok=True)
    
    # Generate all paper sections
    await methodology_composing(research_field, instance_id)
    await related_work_composing(research_field, instance_id)
    await experiments_composing(research_field, instance_id)
    await introduction_composing(research_field, instance_id)
    await conclusion_composing(research_field, instance_id)
    await abstract_composing(research_field, instance_id)

    print("Setting up LaTeX project...")
    # Copy style files and template in correct order
    copy_style_files(target_folder)
    create_main_latex_template(target_folder, research_field, instance_id)
    create_bibliography_file(target_folder)
    
    # Clean and process tex files
    clean_tex_files_in_folder(target_folder)
    
    tex_file_path = os.path.join(target_folder, 'related_work.tex')
    bib_file_path = os.path.join(target_folder, 'iclr2025_conference.bib')
    process_tex_file(tex_file_path, bib_file_path)

    # Compile LaTeX to PDF
    main_file = "iclr2025_conference.tex"
    compile_latex_project(target_folder, main_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_field", type=str, required=True)
    parser.add_argument("--instance_id", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(writing(args.research_field, args.instance_id))