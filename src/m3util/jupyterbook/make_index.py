import os
import argparse
import sys
import nbformat
import re


def extract_first_heading(file_path):
    """
    Extract the first heading from a markdown or notebook file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The first heading found in the file, or the file name if no heading is found.
    """
    if file_path.endswith(".md") and file_path != "index.md":
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    return line.strip("# ").strip()
    elif file_path.endswith(".ipynb"):
        with open(file_path, "r") as f:
            nb = nbformat.read(f, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "markdown":
                    for line in cell.source.splitlines():
                        if line.startswith("#"):
                            return line.strip("# ").strip()
    return os.path.basename(file_path).replace(".md", "").replace(".ipynb", "").replace("_", " ").replace("-", " ")


def extract_number_from_title(title):
    """
    Extract the number from the title before the first underscore or hyphen.

    Args:
        title (str): The title of the file.

    Returns:
        int: The extracted number, or a large number if no number is found.
    """
    match = re.match(r"(\d+)[_-]", title)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number if no number is found


def generate_index(folder_path, output_file, title, start_number):
    """
    Generate an index.md file for a JupyterBook with a single session.

    Args:
        folder_path (str): The path to the folder containing the markdown and notebook files.
        output_file (str): The name of the output index file.
        title (str): Custom title for the session.
        start_number (int): Starting number for the file entries.
    """
    # Get a list of .md and .ipynb files in the folder, excluding index.md
    files = [
        f for f in os.listdir(folder_path)
        if (f.endswith(".md") or f.endswith(".ipynb")) and f != "index.md"
    ]

    # Sort files by the number in the title before the first underscore or hyphen
    files.sort(key=lambda f: extract_number_from_title(os.path.splitext(f)[0]))

    # Generate index content
    content = [f"# {title}"]
    for i, file in enumerate(files, start=start_number):
        file_path = os.path.join(folder_path, file)
        doc_title = extract_first_heading(file_path)
        file_name = os.path.splitext(file)[0]  # Remove the file extension
        content.append(f"{i}. {{doc}}`{doc_title} <./{file_name}>`")

    # Write to the index.md file in the specified folder
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, "w") as f:
        f.write("\n".join(content))

    print(f"Index file '{output_path}' created successfully.")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate an index.md file for a JupyterBook with a single session."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing the markdown and notebook files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="index.md",
        help="Name of the output index file (default: index.md).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Session",
        help="Custom title for the session (default: 'Session').",
    )
    parser.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Starting number for the file entries (default: 1).",
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the function with arguments
    generate_index(args.folder_path, args.output_file, args.title, args.start_number)


if __name__ == "__main__":
    sys.exit(main())
