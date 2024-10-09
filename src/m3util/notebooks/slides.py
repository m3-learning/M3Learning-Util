import json
import argparse


def convert_notebook_to_slides(
    notebook_path, slide_level, subslide_level, output_path=None
):
    """
    Convert a Jupyter notebook to slides by modifying the metadata of each cell.

    Parameters:
    notebook_path (str): Path to the Jupyter notebook (.ipynb) file.
    slide_level (int): Heading level for slides.
    subslide_level (int): Heading level for subslides.
    output_path (str, optional): Output file path. Defaults to None, which appends '_slides' to the original filename.

    Returns:
    None
    """
    # Load the Jupyter notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Process each cell to determine the slide type based on headings
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and cell["source"]:
            # Identify the heading level
            first_line = cell["source"][0]
            if first_line.startswith("#" * slide_level + " "):  # Slide level
                cell["metadata"]["slideshow"] = {"slide_type": "slide"}
            elif first_line.startswith("#" * subslide_level + " "):  # Subslide level
                cell["metadata"]["slideshow"] = {"slide_type": "subslide"}
            else:
                cell["metadata"]["slideshow"] = {"slide_type": "fragment"}
        else:
            # Non-markdown cells default to fragment
            cell["metadata"]["slideshow"] = {"slide_type": "fragment"}

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
        description="Convert a Jupyter notebook to slides."
    )
    parser.add_argument(
        "notebook_path", help="Path to the Jupyter notebook (.ipynb) file"
    )
    parser.add_argument(
        "--slide_level",
        type=int,
        default=1,
        help="Heading level for slides (default: 1)",
    )
    parser.add_argument(
        "--subslide_level",
        type=int,
        default=3,
        help="Heading level for subslides (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: original path with '_slides')",
    )
    
    args = parser.parse_args()
    
    convert_notebook_to_slides(
        notebook_path=args.notebook_path,
        slide_level=args.slide_level,
        subslide_level=args.subslide_level,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()
