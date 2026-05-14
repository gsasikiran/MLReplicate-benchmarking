"""
Integration helper to update legacy code to use centralized directory management.
This module provides backward-compatible functions and migration utilities.

Usage:
    from integration_helper import get_workplace_paper_path, ensure_run_initialized
    
    # Ensure a run is active
    ensure_run_initialized(title="Research Run", reference="Paper ABC")
    
    # Get legacy-compatible paths
    path = get_workplace_paper_path(instance_id, model)
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from directory_manager import get_directory_manager

class LegacyPathAdapter:
    """Adapter to handle legacy path patterns with new centralized system."""
    
    def __init__(self):
        self.manager = get_directory_manager()
        if not self.manager.current_run_dir:
            run_dir_env = os.getenv('AI_RESEARCHER_RUN_DIR')
            if run_dir_env:
                from pathlib import Path
                self.manager.current_run_dir = Path(run_dir_env).absolute()  # ADD .absolute()

    def get_workplace_paper_path(self, instance_id: str, model: str) -> str:
        """
        Replace legacy: workplace_paper/task_{instance_id}_{model}
        With centralized: {current_run}/workplace/
        """
        workplace_dir = self.manager.get_workplace_dir()
        task_dir = workplace_dir / f"task_{instance_id}_{model.replace('/', '__')}"
        task_dir.mkdir(parents=True, exist_ok=True)
        return str(task_dir)
    
    def get_cache_path(self, instance_id: str, model: str) -> str:
        """
        Replace legacy: cache_{instance_id}_{model}
        With centralized: {current_run}/cache/cache_{instance_id}_{model}
        """
        cache_dir = self.manager.get_cache_dir()
        cache_subdir = cache_dir / f"cache_{instance_id}_{model.replace('/', '__')}"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return str(cache_subdir)
    
    def get_paper_agent_path(self, research_field: str, instance_id: str) -> str:
        """
        Replace legacy: paper_agent/{research_field}/{instance_id}
        With centralized: {current_run}/paper/{research_field}_{instance_id}
        """
        paper_dir = self.manager.get_paper_dir()
        paper_subdir = paper_dir / f"{research_field}_{instance_id}"
        paper_subdir.mkdir(parents=True, exist_ok=True)
        return str(paper_subdir)
    
    def get_logs_path(self, component: str = "main") -> str:
        """
        Centralized logs directory with component-specific subdirectories.
        """
        logs_dir = self.manager.get_logs_dir()
        component_dir = logs_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        return str(component_dir)

    def migrate_existing_results(self, old_path: str, target_subdir: str = "") -> Optional[str]:
        """
        Migrate existing results from old scattered locations to centralized structure.
        
        Args:
            old_path: Path to existing results
            target_subdir: Target subdirectory in the new structure
            
        Returns:
            New path if migration successful, None otherwise
        """
        if not os.path.exists(old_path):
            return None
            
        try:
            new_path = self.manager.copy_to_run_dir(old_path, target_subdir)
            return str(new_path)
        except Exception as e:
            print(f"Migration failed for {old_path}: {e}")
            return None


# Global adapter instance
_legacy_adapter = None

def get_legacy_adapter() -> LegacyPathAdapter:
    """Get the global legacy path adapter."""
    global _legacy_adapter
    if _legacy_adapter is None:
        _legacy_adapter = LegacyPathAdapter()
    return _legacy_adapter

# Backward-compatible functions for easy migration

def get_workplace_paper_path(instance_id: str, model: str) -> str:
    """Backward-compatible function for research_agent."""
    return get_legacy_adapter().get_workplace_paper_path(instance_id, model)

def get_cache_path(instance_id: str, model: str) -> str:
    """Backward-compatible function for caching."""
    return get_legacy_adapter().get_cache_path(instance_id, model)

def get_paper_agent_path(research_field: str, instance_id: str) -> str:
    """Backward-compatible function for paper_agent."""
    return get_legacy_adapter().get_paper_agent_path(research_field, instance_id)

def get_component_logs_path(component: str = "main") -> str:
    """Backward-compatible function for logging."""
    return get_legacy_adapter().get_logs_path(component)

def ensure_run_initialized(title: str = None, reference: str = None) -> Path:
    """Ensure a run is initialized, using existing one if available."""
    manager = get_directory_manager()
    return manager.initialize_run(title, reference) 

# Migration utilities

def migrate_research_results(research_field: str, instance_id: str, model: str):
    """Migrate existing research results to centralized structure."""
    adapter = get_legacy_adapter()
    
    # List of common old paths to check and migrate
    old_paths = [
        f"research_agent/cache_{instance_id}_{model}",
        f"research_agent/workplace_paper/task_{instance_id}_{model}",
        f"paper_agent/{research_field}/{instance_id}",
        "logs",
        "casestudy_results",
    ]
    
    migrated_paths = {}
    
    for old_path in old_paths:
        if os.path.exists(old_path):
            # Determine target directory based on content type
            if "cache" in old_path:
                target_subdir = "cache"
            elif "workplace" in old_path:
                target_subdir = "workplace"
            elif "paper_agent" in old_path:
                target_subdir = "paper"
            elif "logs" in old_path:
                target_subdir = "logs"
            else:
                target_subdir = "misc"
                
            new_path = adapter.migrate_existing_results(old_path, target_subdir)
            if new_path:
                migrated_paths[old_path] = new_path
                print(f"Migrated: {old_path} -> {new_path}")
    
    return migrated_paths

def cleanup_old_scattered_files():
    """Clean up old scattered files after successful migration."""
    # Only clean up if we have a current run (indicating successful migration)
    manager = get_directory_manager()
    if manager.get_current_run_directory() is None:
        print("No current run found - skipping cleanup")
        return
    
    # Remove __pycache__ directories
    manager.clean_pycache()
    
    # List of old directories that can be safely removed after migration
    old_dirs = [
        "research_agent/cache_*",
        "research_agent/workplace_paper",
        "casestudy_results",
    ]
    
    for pattern in old_dirs:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    print(f"Cleaned up old directory: {path}")
                except Exception as e:
                    print(f"Could not clean up {path}: {e}")

# Test function to verify the new system
def test_new_system():
    """Test function to verify the new system works correctly."""
    print("Testing new centralized directory system...")
    
    try:
        # Test creating a new run
        run_dir = initialize_run(title="Test Migration", reference="Migration Test")
        print(f"✓ Created run directory: {run_dir}")
        
        # Test getting standard directories
        research_dir = get_legacy_adapter().manager.get_research_dir()
        paper_dir = get_legacy_adapter().manager.get_paper_dir()
        logs_dir = get_legacy_adapter().manager.get_logs_dir()
        
        print(f"✓ Research directory: {research_dir}")
        print(f"✓ Paper directory: {paper_dir}")
        print(f"✓ Logs directory: {logs_dir}")
        
        # Test legacy adapter functions
        workplace_path = get_workplace_paper_path("test_instance", "gpt-4o")
        cache_path = get_cache_path("test_instance", "gpt-4o")
        
        print(f"✓ Workplace path: {workplace_path}")
        print(f"✓ Cache path: {cache_path}")
        
        # Test file saving
        from directory_manager import save_to_run
        test_file = save_to_run("Test content", "test.txt", "logs")
        print(f"✓ Saved test file: {test_file}")
        
        print("✓ All tests passed! New system is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False