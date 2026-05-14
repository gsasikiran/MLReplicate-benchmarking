#!/usr/bin/env python3
"""
Test DrawerTool to verify it works correctly
"""

import json
import os
from tiny_scientist.tool import DrawerTool

def test_drawer():
    """Test if DrawerTool works with correct input format"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print(" ERROR: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        return False
    
    print("=" * 60)
    print("Testing DrawerTool")
    print("=" * 60)
    
    method_text = """
    We propose a novel approach to handle missing data in score matching.
    Our method uses importance weighting to account for missing values.
    We also employ variational inference to estimate the score function.
    The algorithm consists of three main steps: imputation, weighting, and optimization.
    """
    
    print("\nTest 1: Creating DrawerTool instance...")
    try:
        drawer = DrawerTool(
            model="gpt-4o-mini",
            temperature=0.75
        )
        print("✓ DrawerTool created successfully")
    except Exception as e:
        print(f"✗ Failed to create DrawerTool: {e}")
        return False
    
    print("\nTest 2: Testing with double-encoded JSON (prevent MCP parsing)...")
    try:
        inner_json = json.dumps({
            "section_name": "Method",
            "section_content": method_text
        })
        query = json.dumps(inner_json) 
        
        print(f"  Query type: {type(query)}")
        print(f"  Query is string: {isinstance(query, str)}")
        print(f"  Query preview: {query[:150]}...")
        
        print("\n  Calling drawer.run()...")
        result = drawer.run(query)
        
        print("✓ Drawer call succeeded!")
        print(f"  Result type: {type(result)}")
        print(f"  Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        
        if result and isinstance(result, dict):
            if "diagram" in result:
                diagram = result["diagram"]
                print(f"  Diagram keys: {diagram.keys() if isinstance(diagram, dict) else 'N/A'}")
                
                if diagram.get("svg"):
                    print(f"  ✓ SVG generated ({len(diagram['svg'])} chars)")
                if diagram.get("summary"):
                    print(f"  ✓ Summary: {diagram['summary'][:100]}...")
            else:
                print(" No 'diagram' key in result")
                print(f"  Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Drawer call failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    success = test_drawer()
    
    if success:
        print("\n All tests passed! DrawerTool is working correctly.")
        exit(0)
    else:
        print("\n Tests failed! Check the errors above.")
        exit(1)