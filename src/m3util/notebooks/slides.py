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


def has_iframe(source):
    """
    Check if a code cell's source contains an iframe.
    """
    if isinstance(source, list):
        source = "".join(source)
    return "IFrame" in source.lower()


def convert_notebook_to_slides(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to slides by modifying the metadata of each cell
    and splitting cells at headings.

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
            # Split markdown cells at headings
            split_cells = split_cell_at_headings(cell)
            new_cells.extend(split_cells)
        else:
            # Non-markdown cells become fragments
            cell["metadata"]["slideshow"] = {"slide_type": "fragment"}

            # Add hide-input tag for code cells with iframes
            if cell["cell_type"] == "code" and has_iframe(cell["source"]):
                if "tags" not in cell["metadata"]:
                    cell["metadata"]["tags"] = []
                if "hide-input" not in cell["metadata"]["tags"]:
                    cell["metadata"]["tags"].append("hide-input")

            new_cells.append(cell)

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
        "and everything else into fragments. Adds hide-input tag to code cells with iframes."
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
