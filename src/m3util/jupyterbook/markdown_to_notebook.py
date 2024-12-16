import re
import sys
import os
import nbformat as nbf


def markdown_to_notebook(input_file: str):
    """
    Converts a Markdown file into a Jupyter notebook.

    Args:
        input_file (str): Path to the Markdown file.
    """
    # Derive the output file name by replacing the extension with `.ipynb`
    output_file = os.path.splitext(input_file)[0] + ".ipynb"

    # Read the Markdown file
    with open(input_file, "r", encoding="utf-8") as file:
        markdown_lines = file.readlines()

    # Initialize a new Jupyter notebook
    notebook = nbf.v4.new_notebook()
    cells = []
    code_block = False
    code_lines = []

    for line in markdown_lines:
        # Check for start of a Python code block
        if line.strip().startswith("```python"):
            code_block = True
            continue
        # Check for end of a code block
        elif line.strip() == "```" and code_block:
            cells.append(nbf.v4.new_code_cell("\n".join(code_lines)))
            code_block = False
            code_lines = []
            continue

        if code_block:
            # Collect lines for a code cell
            code_lines.append(line.rstrip())
        else:
            # Check if the line is a heading or a numbered list item
            if line.strip().startswith("#") or re.match(r"^\d+\.\s", line.strip()):
                cells.append(nbf.v4.new_markdown_cell(line.strip()))
            else:
                # Non-heading/non-numbered lines are Markdown content
                if cells and cells[-1].cell_type == "markdown":
                    # Append to the last Markdown cell if possible
                    cells[-1].source += f"\n{line.strip()}"
                else:
                    # Start a new Markdown cell
                    cells.append(nbf.v4.new_markdown_cell(line.strip()))

    # Add the collected cells to the notebook
    notebook.cells = cells

    # Write the notebook to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        nbf.write(notebook, file)

    print(f"Notebook saved to: {output_file}")


def main(args=None):
    """
    Entry point for the script.
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: md_to_notebook <input_markdown_file>")
        sys.exit(1)

    input_file = args[0]

    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        sys.exit(1)

    try:
        markdown_to_notebook(input_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
