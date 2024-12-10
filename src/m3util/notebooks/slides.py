import json
import argparse


def split_cell_at_headings(cell):
    """
    Split a markdown cell into multiple cells whenever a heading is encountered.
    Returns a list of new cells with appropriate slide types.
    """
    if cell["cell_type"] != "markdown" or not cell["source"]:
        return [cell]

    new_cells = []
    current_content = []
    current_type = "fragment"  # Default type is now fragment

    for line in cell["source"]:
        # Check if line is a heading
        if line.startswith(("# ", "## ", "### ")):
            # Save the previous content if there is any
            if current_content:
                new_cell = {
                    "cell_type": "markdown",
                    "metadata": {"slideshow": {"slide_type": current_type}},
                    "source": current_content,
                }
                new_cells.append(new_cell)

            # Start new content with the heading
            current_content = [line]
            # Set slide type based on heading level
            if line.startswith(("# ", "## ")):
                current_type = "slide"
            elif line.startswith("### "):
                current_type = "subslide"
            else:
                current_type = "fragment"
        else:
            current_content.append(line)

    # Add the last cell if there's remaining content
    if current_content:
        new_cell = {
            "cell_type": "markdown",
            "metadata": {"slideshow": {"slide_type": current_type}},
            "source": current_content,
        }
        new_cells.append(new_cell)

    return new_cells


def extract_code_blocks(cell):
    """
    Extract explicitly defined code blocks (` ```python `) from a markdown cell
    into separate code cells.
    """
    if cell["cell_type"] != "markdown":
        return [cell]

    new_cells = []
    current_markdown = []
    inside_code_block = False
    code_block = []

    for line in cell["source"]:
        if line.strip().startswith("```python") and not inside_code_block:
            # Start of an explicit code block
            inside_code_block = True
            if current_markdown:
                new_cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": current_markdown,
                    }
                )
                current_markdown = []
            code_block = []
        elif inside_code_block and line.strip() == "```":
            # End of an explicit code block
            inside_code_block = False
            new_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": code_block,
                }
            )
            code_block = []
        elif inside_code_block:
            code_block.append(line)
        else:
            current_markdown.append(line)

    # Add any remaining markdown
    if current_markdown:
        new_cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": current_markdown,
            }
        )

    return new_cells


def has_iframe(source):
    """
    Check if a code cell's source contains an iframe.
    """
    if isinstance(source, list):
        source = "".join(source)
    return "IFrame" in source.lower()


def remove_empty_markdown_cells(cells):
    """
    Remove all empty markdown cells from the notebook.
    """
    return [
        cell
        for cell in cells
        if not (
            cell["cell_type"] == "markdown"
            and not "".join(cell.get("source", [])).strip()
        )
    ]


def convert_notebook_to_slides(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to slides by modifying the metadata of each cell
    and splitting cells at headings and code blocks.

    Parameters:
    notebook_path (str): Path to the Jupyter notebook (.ipynb) file.
    output_path (str, optional): Output file path. Defaults to None, which uses the original filename.

    Returns:
    None
    """
    # Load the Jupyter notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Process each cell and collect new cells
    new_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            # Extract code blocks first
            extracted_cells = extract_code_blocks(cell)
            # Then split at headings for remaining markdown cells
            for extracted_cell in extracted_cells:
                split_cells = split_cell_at_headings(extracted_cell)
                new_cells.extend(split_cells)
        else:
            # For non-markdown cells
            cell["metadata"]["slideshow"] = {"slide_type": "fragment"}

            # Add hide-input tag for code cells with iframes
            if cell["cell_type"] == "code" and has_iframe(cell["source"]):
                if "tags" not in cell["metadata"]:
                    cell["metadata"]["tags"] = []
                if "hide-input" not in cell["metadata"]["tags"]:
                    cell["metadata"]["tags"].append("hide-input")

            new_cells.append(cell)

    # Remove all empty markdown cells
    new_cells = remove_empty_markdown_cells(new_cells)

    # Update notebook with new cells
    notebook["cells"] = new_cells

    # Define output path
    if output_path is None:
        output_path = notebook_path

    # Save the modified notebook back to a file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook converted to slides and saved as {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert a Jupyter notebook to slides. Splits cells at headings, "
        "makes level 1 and 2 headings into slides, level 3 into subslides, "
        "and everything else into fragments. Extracts all Python code from markdown cells."
    )
    parser.add_argument(
        "notebook_path", help="Path to the Jupyter notebook (.ipynb) file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: original path)",
    )
    args = parser.parse_args()

    convert_notebook_to_slides(
        notebook_path=args.notebook_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
