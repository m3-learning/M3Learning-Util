import nbformat
import argparse
import sys


# Function to extract Python blocks from Markdown cells and insert them as Python cells
def extract_markdown_python_blocks(notebook_path, output_notebook_path=None):
    try:
        # Load the Jupyter Notebook
        with open(notebook_path, "r") as nb_file:
            notebook = nbformat.read(nb_file, as_version=4)

        new_cells = []

        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                # Check for Python code blocks in Markdown
                lines = cell.source.splitlines()
                in_code_block = False
                code_block = []
                markdown_content = []

                for line in lines:
                    if line.strip().startswith("```python"):
                        in_code_block = True
                        # Finish the current Markdown cell
                        if markdown_content:
                            new_cells.append(
                                nbformat.v4.new_markdown_cell(
                                    source="\n".join(markdown_content)
                                )
                            )
                            markdown_content = []
                    elif line.strip() == "```" and in_code_block:
                        in_code_block = False
                        # Add extracted code block as a new code cell
                        new_cells.append(
                            nbformat.v4.new_code_cell(source="\n".join(code_block))
                        )
                        code_block = []
                    elif in_code_block:
                        code_block.append(line)
                    else:
                        markdown_content.append(line)

                # Add remaining Markdown content as a cell
                if markdown_content:
                    new_cells.append(
                        nbformat.v4.new_markdown_cell(
                            source="\n".join(markdown_content)
                        )
                    )
            else:
                # Retain other cell types as is
                new_cells.append(cell)

        # Replace the cells in the notebook with the new set of cells
        notebook.cells = new_cells

        # Save the updated notebook, replacing the original if output_notebook_path is None
        save_path = output_notebook_path if output_notebook_path else notebook_path
        with open(save_path, "w") as output_file:
            nbformat.write(notebook, output_file)

        print(
            f"Notebook successfully updated with extracted Python blocks at: {save_path}"
        )
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to parse arguments and process the notebook.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Extract Python blocks from Markdown cells and insert them as code cells."
    )
    parser.add_argument(
        "notebook_path",
        type=str,
        help="Path to the input Jupyter Notebook.",
    )
    parser.add_argument(
        "output_notebook_path",
        nargs="?",
        default=None,
        help="Path to save the updated Jupyter Notebook. If not provided, the original notebook will be replaced.",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Process the notebook
        extract_markdown_python_blocks(args.notebook_path, args.output_notebook_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
