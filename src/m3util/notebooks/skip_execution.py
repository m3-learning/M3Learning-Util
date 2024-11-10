# src/m3util/notebooks/skip-execution.py
import json
import sys
from pathlib import Path

def add_tags_to_notebook(notebook_path):
    """
    Add skip-execution tags to all cells in a Jupyter notebook.
    
    Args:
        notebook_path (str): Path to the Jupyter notebook file
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Track if any changes were made
    changes_made = False
    
    # Add tags to each cell
    for cell in notebook['cells']:
        if 'metadata' not in cell:
            cell['metadata'] = {}
            changes_made = True
        if 'tags' not in cell['metadata']:
            cell['metadata']['tags'] = []
            changes_made = True
        
        # Add tags if they don't already exist
        tags_to_add = ['skip-execution']
        for tag in tags_to_add:
            if tag not in cell['metadata']['tags']:
                cell['metadata']['tags'].append(tag)
                changes_made = True
    
    # Write the modified notebook back to the original file
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"Tags added successfully to: {notebook_path}")
    else:
        print(f"All required tags already present in: {notebook_path}")

def main(args=None):
    """
    Entry point for the notebook tagger.
    """
    if args is None:
        args = sys.argv[1:]
    
    if len(args) != 1:
        print("Usage: skip-execution <notebook_path>")
        sys.exit(1)
    
    notebook_path = args[0]
    add_tags_to_notebook(notebook_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())