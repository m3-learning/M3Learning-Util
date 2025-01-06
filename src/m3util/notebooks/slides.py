import json
import argparse
import os


def split_cell_at_headings(cell):
    # Code as provided in the original script
    if cell["cell_type"] != "markdown" or not cell["source"]:
        return [cell]

    new_cells = []
    current_content = []
    current_type = "fragment"  # Default type is now fragment

    for line in cell["source"]:
        if line.startswith(("# ", "## ", "### ")):
            if current_content:
                new_cell = {
                    "cell_type": "markdown",
                    "metadata": {"slideshow": {"slide_type": current_type}},
                    "source": current_content,
                }
                new_cells.append(new_cell)
            current_content = [line]
            if line.startswith(("# ", "## ")):
                current_type = "slide"
            elif line.startswith("### "):
                current_type = "subslide"
            else:
                current_type = "fragment"
        else:
            current_content.append(line)

    if current_content:
        new_cell = {
            "cell_type": "markdown",
            "metadata": {"slideshow": {"slide_type": current_type}},
            "source": current_content,
        }
        new_cells.append(new_cell)

    return new_cells


def extract_code_blocks(cell):
    # Code as provided in the original script
    if cell["cell_type"] != "markdown":
        return [cell]

    new_cells = []
    current_markdown = []
    inside_code_block = False
    code_block = []

    for line in cell["source"]:
        if line.strip().startswith("```python") and not inside_code_block:
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
    if isinstance(source, list):
        source = "".join(source)
    return "IFrame" in source.lower()


def remove_empty_markdown_cells(cells):
    return [
        cell
        for cell in cells
        if not (
            cell["cell_type"] == "markdown"
            and not "".join(cell.get("source", [])).strip()
        )
    ]


def convert_notebook_to_slides(notebook_path, output_path=None):
    # Code as provided in the original script
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    new_cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            extracted_cells = extract_code_blocks(cell)
            for extracted_cell in extracted_cells:
                split_cells = split_cell_at_headings(extracted_cell)
                new_cells.extend(split_cells)
        else:
            cell["metadata"]["slideshow"] = {"slide_type": "fragment"}

            if cell["cell_type"] == "code" and has_iframe(cell["source"]):
                if "tags" not in cell["metadata"]:
                    cell["metadata"]["tags"] = []
                if "hide-input" not in cell["metadata"]["tags"]:
                    cell["metadata"]["tags"].append("hide-input")

            new_cells.append(cell)

    new_cells = remove_empty_markdown_cells(new_cells)

    notebook["cells"] = new_cells

    if output_path is None:
        output_path = notebook_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook converted to slides and saved as {output_path}")


def process_folder(folder_path, output_folder=None):
    """
    Process all .ipynb files in a folder and convert them to slides.
    Parameters:
    folder_path (str): Path to the folder containing .ipynb files.
    output_folder (str, optional): Path to save the converted files. Defaults to None (overwrite original).
    """
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".ipynb"):
            notebook_path = os.path.join(folder_path, filename)
            if output_folder:
                output_path = os.path.join(output_folder, filename)
            else:
                output_path = None
            convert_notebook_to_slides(notebook_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to slides for all files in a folder."
    )
    parser.add_argument(
        "folder_path", help="Path to the folder containing .ipynb files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Folder to save the converted files (default: overwrite original files)",
    )
    args = parser.parse_args()

    process_folder(
        folder_path=args.folder_path,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()
