import re
import sys
import json
import re


def remove_bold_markers_from_headings(input_file: str):
    """
    Reads a Jupyter Notebook file, removes bold markers (** or __) from text
    in Markdown cells, and overwrites the file.

    Args:
        input_file (str): Path to the input Jupyter Notebook file.
    """
    # Load the notebook
    with open(input_file, "r", encoding="utf-8") as file:
        notebook = json.load(file)

    # Regex pattern to match and remove bold markers in Markdown text
    bold_pattern = re.compile(r"(\*\*|__)(.*?)(\*\*|__)")

    # Process only Markdown cells
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":  # Only process Markdown cells
            modified_source = []
            for line in cell.get("source", []):
                # Remove bold markers from each line
                modified_line = bold_pattern.sub(r"\2", line)
                modified_source.append(modified_line)
            cell["source"] = modified_source  # Update the cell source

    # Overwrite the input file with the modified content
    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(notebook, file, indent=2, ensure_ascii=False)


def main(args=None):
    """
    Entry point for the script.
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: remove-bold-markers <input_file>")
        sys.exit(1)

    input_file = args[0]

    try:
        remove_bold_markers_from_headings(input_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
