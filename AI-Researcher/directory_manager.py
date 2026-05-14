import os
import glob
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
from research_agent.inno.util import single_select_menu
from typing import List, Optional, Tuple

class DirectoryManager:
    """Centralized directory management for AI-Researcher results."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the directory manager.
        
        Args:
            base_dir: Base directory for the project. Defaults to current working directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.results_dir = self.base_dir / "results"
        self.current_run_dir = None
        self.run_metadata = {}
    
    def _detect_incomplete_runs(self) -> list:
        """Detect runs that might be incomplete (no completion marker)."""
        incomplete_runs = []
        if not self.results_dir.exists():
            return incomplete_runs
            
        for run_dir in sorted(self.results_dir.glob("test_*"), reverse=True):
            if run_dir.is_dir():
                # Check if run has completion marker
                completion_marker = run_dir / ".completed"
                if not completion_marker.exists():
                    incomplete_runs.append(run_dir)
        
        return incomplete_runs
    
    def _prompt_resume_or_new(self, incomplete_runs: List[Path]) -> Tuple[bool, Optional[Path]]:
        """Prompt user to resume or start new run"""
        if not incomplete_runs:
            return False, None
        
        latest_run = incomplete_runs[0]
        
        # Print incomplete run info
        print(f"\nFound incomplete run: {latest_run.name}")
        metadata_file = latest_run / "run_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                print(f"  Title: {metadata.get('title', 'N/A')}")
                print(f"  Started: {metadata.get('start_time', 'N/A')}")
            except:
                pass
        
        # Check for AUTO_RESUME environment variable
        auto_resume = os.getenv("AUTO_RESUME", "").lower()
        if auto_resume in ["new", "start_new", "false", "no"]:
            print("AUTO_RESUME=new detected, starting new run")
            return False, None
        elif auto_resume in ["resume", "true", "yes"]:
            print("AUTO_RESUME=resume detected, resuming existing run")
            return True, latest_run
        
        # Interactive prompt (only if not in batch mode)
        choice = single_select_menu(
            ["Resume", "Start New"], 
            "Do you want to resume this run or start a new one?"
        )
        
        if choice == "Resume":
            return True, latest_run
        else:
            return False, None
    
    def initialize_run(self, title: str = None, reference: str = None) -> Path:
        """Initialize a run with resume capability."""
        self.results_dir.mkdir(exist_ok=True)
        
        # Check for incomplete runs
        incomplete_runs = self._detect_incomplete_runs()
        should_resume, resume_dir = self._prompt_resume_or_new(incomplete_runs)
        
        if should_resume and resume_dir:
            # Resume existing run
            self.current_run_dir = resume_dir
            print(f"Resuming run: {resume_dir.name}")
            
            # Load existing metadata
            metadata_file = resume_dir / "run_metadata.json"
            if metadata_file.exists():
                try:
                    self.run_metadata = json.loads(metadata_file.read_text())
                except:
                    self.run_metadata = {"run_name": resume_dir.name}
            else:
                self.run_metadata = {"run_name": resume_dir.name}
        else:
            # Create new run
            next_num = self._get_next_test_number()
            run_name = f"test_{next_num:03d}"
            
            self.current_run_dir = self.results_dir / run_name
            self.current_run_dir.mkdir(exist_ok=True)
            
            self.run_metadata = {
                "run_name": run_name,
                "title": title,
                "reference": reference,
                "start_time": datetime.now().isoformat()
            }
            
            # Save metadata
            metadata_file = self.current_run_dir / "run_metadata.json"
            metadata_file.write_text(json.dumps(self.run_metadata, indent=2))
            
            print(f"Created new run: {run_name}")
        
        # Ensure subdirectories exist
        self._create_standard_subdirectories()
        return self.current_run_dir
    
    def mark_run_completed(self):
        """Mark the current run as completed."""
        if self.current_run_dir:
            completion_marker = self.current_run_dir / ".completed"
            completion_marker.write_text(datetime.now().isoformat())
    
    def create_new_run_directory(self, title: str = None, reference: str = None) -> Path:
        """Create a new auto-incrementing results directory.
        
        Args:
            title: Optional title for the run
            reference: Optional reference for the run
            
        Returns:
            Path to the created directory
        """
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Find next available test number
        next_num = self._get_next_test_number()
        run_name = f"test_{next_num:03d}"
        
        # Create the new run directory
        self.current_run_dir = self.results_dir / run_name
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create metadata
        self.run_metadata = {
            "run_name": run_name,
            "title": title,
            "reference": reference,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_dir": str(self.base_dir),
            "run_dir": str(self.current_run_dir)
        }
        
        # Save metadata
        self._save_run_metadata()
        
        # Create standard subdirectories
        self._create_standard_subdirectories()
        
        return self.current_run_dir
    
    def get_current_run_directory(self) -> Optional[Path]:
        """Get the current run directory."""
        return self.current_run_dir
    
    def get_research_dir(self) -> Path:
        """Get the research agent output directory."""
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        return self.current_run_dir / "research"
    
    def get_paper_dir(self) -> Path:
        """Get the paper agent output directory.""" 
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        return self.current_run_dir / "paper"
    
    def get_logs_dir(self) -> Path:
        """Get the logs directory."""
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        return self.current_run_dir / "logs"
    
    def get_cache_dir(self) -> Path:
        """Get the cache directory."""
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        return self.current_run_dir / "cache"
    
    def get_workplace_dir(self) -> Path:
        """Get the workplace directory."""
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        return self.current_run_dir / "workplace"
    
    def clean_pycache(self):
        """Remove all __pycache__ directories from the current run."""
        if not self.current_run_dir:
            return
            
        for pycache_dir in self.current_run_dir.rglob("__pycache__"):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir, ignore_errors=True)
    
    def save_file(self, content: str, filename: str, subdir: str = "") -> Path:
        """Save content to a file in the current run directory.
        
        Args:
            content: Content to save
            filename: Name of the file
            subdir: Optional subdirectory within the run directory
            
        Returns:
            Path to the saved file
        """
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        
        target_dir = self.current_run_dir / subdir if subdir else self.current_run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    def copy_to_run_dir(self, source_path: str, target_subdir: str = "") -> Path:
        """Copy a file or directory to the current run directory.
        
        Args:
            source_path: Source file or directory path
            target_subdir: Optional subdirectory within the run directory
            
        Returns:
            Path to the copied file/directory
        """
        if not self.current_run_dir:
            raise RuntimeError("No run directory created. Call initialize_run() first.")
        
        source = Path(source_path)
        target_dir = self.current_run_dir / target_subdir if target_subdir else self.current_run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / source.name
        
        if source.is_file():
            shutil.copy2(source, target_path)
        elif source.is_dir():
            shutil.copytree(source, target_path, dirs_exist_ok=True)
        
        return target_path
    
    def list_all_runs(self) -> Dict[str, Dict[str, Any]]:
        """List all existing runs with their metadata.
        
        Returns:
            Dictionary mapping run names to their metadata
        """
        runs = {}
        if not self.results_dir.exists():
            return runs
            
        for run_dir in sorted(self.results_dir.glob("test_*")):
            if run_dir.is_dir():
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        runs[run_dir.name] = metadata
                    except (json.JSONDecodeError, IOError):
                        # Fallback metadata if file is corrupted
                        runs[run_dir.name] = {
                            "run_name": run_dir.name,
                            "created_at": "unknown",
                            "title": "unknown",
                            "reference": "unknown"
                        }
        return runs
    
    def cleanup_old_runs(self, keep_count: int = 10):
        """Clean up old runs, keeping only the most recent ones.
        
        Args:
            keep_count: Number of runs to keep
        """
        if not self.results_dir.exists():
            return
            
        all_runs = sorted(self.results_dir.glob("test_*"), key=lambda x: x.name)
        if len(all_runs) <= keep_count:
            return
            
        # Remove oldest runs
        runs_to_remove = all_runs[:-keep_count]
        for run_dir in runs_to_remove:
            shutil.rmtree(run_dir, ignore_errors=True)
    
    def _get_next_test_number(self) -> int:
        """Get the next available test number."""
        if not self.results_dir.exists():
            return 1
            
        existing_tests = list(self.results_dir.glob("test_*"))
        if not existing_tests:
            return 1
            
        max_num = 0
        for test_dir in existing_tests:
            try:
                num = int(test_dir.name.split('_')[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
                
        return max_num + 1
    
    def _create_standard_subdirectories(self):
        """Create standard subdirectories in the run directory."""
        subdirs = ["research", "paper", "logs", "cache", "workplace"]
        for subdir in subdirs:
            (self.current_run_dir / subdir).mkdir(exist_ok=True)
    
    def _save_run_metadata(self):
        """Save run metadata to JSON file."""
        if self.current_run_dir:
            metadata_file = self.current_run_dir / "run_metadata.json"
            metadata_file.write_text(json.dumps(self.run_metadata, indent=2))


# Global instance for easy access
_global_directory_manager = None

def get_directory_manager() -> DirectoryManager:
    """Get the global directory manager instance."""
    global _global_directory_manager
    if _global_directory_manager is None:
        _global_directory_manager = DirectoryManager()
    return _global_directory_manager

# Convenience functions for common directory access
def get_research_dir() -> Path:
    """Get the research directory for the current run."""
    return get_directory_manager().get_research_dir()

def get_paper_dir() -> Path:
    """Get the paper directory for the current run."""
    return get_directory_manager().get_paper_dir()

def get_logs_dir() -> Path:
    """Get the logs directory for the current run."""
    return get_directory_manager().get_logs_dir()

def get_cache_dir() -> Path:
    """Get the cache directory for the current run."""
    return get_directory_manager().get_cache_dir()

def get_workplace_dir() -> Path:
    """Get the workplace directory for the current run."""
    return get_directory_manager().get_workplace_dir()

def save_to_run(content: str, filename: str, subdir: str = "") -> Path:
    """Save content to the current run directory."""
    return get_directory_manager().save_file(content, filename, subdir)

def cleanup_pycache():
    """Clean up all __pycache__ directories in the current run."""
    get_directory_manager().clean_pycache()