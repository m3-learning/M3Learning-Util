import os
import shutil
import nbformat
import subprocess
import sys
import argparse


def process_notebooks(root_folder):
    """
    Recursively checks all files in a folder for Jupyter notebooks containing a specific cell content.
    Moves matching notebooks to a new folder structure and runs `otter assign`.
    """
    # Define target subfolder
    solutions_folder = os.path.join(root_folder, "_solutions")
    os.makedirs(solutions_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                notebook_path = os.path.join(dirpath, filename)

                # Check if the notebook has the target content
                if has_assignment_config(notebook_path):
                    # Create subfolder for the notebook in `_solutions`
                    notebook_name = os.path.splitext(filename)[0]
                    notebook_subfolder = os.path.join(solutions_folder, notebook_name)
                    os.makedirs(notebook_subfolder, exist_ok=True)

                    # Move the notebook to the new subfolder
                    new_notebook_path = os.path.join(notebook_subfolder, filename)
                    if os.path.abspath(notebook_path) != os.path.abspath(
                        new_notebook_path
                    ):
                        shutil.move(notebook_path, new_notebook_path)
                        print(f"Moved: {notebook_path} -> {new_notebook_path}")
                    else:
                        print(f"Notebook already in destination: {new_notebook_path}")

                    # Run `otter assign` on the notebook
                    run_otter_assign(
                        new_notebook_path, os.path.join(notebook_subfolder, "dist")
                    )


def has_assignment_config(notebook_path):
    """
    Checks if a Jupyter notebook contains a cell with the content `# ASSIGNMENT CONFIG`.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "raw" and "# ASSIGNMENT CONFIG" in cell.source:
                    return True
    except Exception as e:
        print(f"Error reading notebook {notebook_path}: {e}")
    return False


def run_otter_assign(notebook_path, dist_folder):
    """
    Runs `otter assign` on the given notebook and outputs to the specified distribution folder.
    """
    try:
        os.makedirs(dist_folder, exist_ok=True)
        command = ["otter", "assign", notebook_path, dist_folder]
        subprocess.run(command, check=True)
        print(f"Otter assign completed: {notebook_path} -> {dist_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error running `otter assign` for {notebook_path}: {e}")
    except Exception as e:
        print(f"Unexpected error during `otter assign` for {notebook_path}: {e}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Recursively process Jupyter notebooks with '# ASSIGNMENT CONFIG', move them to a solutions folder, and run otter assign."
    )
    parser.add_argument(
        "root_folder", type=str, help="Path to the root folder to process"
    )
    args = parser.parse_args()

    process_notebooks(args.root_folder)


if __name__ == "__main__":
    sys.exit(main())
