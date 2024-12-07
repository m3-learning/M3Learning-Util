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

                    student_notebook = os.path.join(
                        notebook_subfolder, "dist", "student", notebook_name + ".ipynb"
                    )

                    clean_notebook(student_notebook)

                    shutil.copy(student_notebook, root_folder)
                    print(
                        f"Copied and cleaned student notebook: {student_notebook} -> {root_folder}"
                    )


def clean_notebook(notebook_path):
    """
    Removes specific cells and makes Markdown cells non-editable and non-deletable by updating their metadata.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        cleaned_cells = []
        for cell in notebook.cells:
            if not hasattr(cell, "cell_type") or not hasattr(cell, "source"):
                continue

            if (
                "## Submission" not in cell.source
                and "# Save your notebook first," not in cell.source
            ):
                # Make Markdown cells non-editable and non-deletable
                if cell.cell_type == "markdown":
                    cell.metadata["editable"] = cell.metadata.get("editable", False)
                    cell.metadata["deletable"] = cell.metadata.get("deletable", False)

                # Add "skip-execution" tag to Code cells
                if cell.cell_type == "code":
                    cell.metadata["tags"] = cell.metadata.get("tags", [])
                    if "skip-execution" not in cell.metadata["tags"]:
                        cell.metadata["tags"].append("skip-execution")

                cleaned_cells.append(cell)
            else:
                print(f"Removed cell: {cell.source.strip()[:50]}...")

        notebook.cells = cleaned_cells

        # Write the updated notebook back to the file
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        print(f"Cleaned notebook: {notebook_path}")

    except Exception as e:
        print(f"Error cleaning notebook {notebook_path}: {e}")


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
