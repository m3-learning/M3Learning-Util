import os
import yaml
import re
import argparse
import sys


def generate_toc(folder_path):
    """
    Generate a _toc.yml file for a JupyterBook folder.

    Args:
        folder_path (str): Path to the folder where _toc.yml will be generated.
    """
    # Validate folder path
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    # Get absolute path and split into components
    abs_path = os.path.abspath(folder_path)
    path_parts = abs_path.split(os.sep)

    # Find index of 'jupyterbook' in path (optional for customization)
    try:
        jb_index = path_parts.index("jupyterbook")
        relative_path = os.sep.join(path_parts[jb_index + 1 :])
    except ValueError:
        relative_path = path_parts[-1]

    # Get files in the folder and sort
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and not f.startswith(".")
        and f != "_toc.yml"
    ]
    files.sort(
        key=lambda x: (
            int(re.match(r"(\d+)", x).group(1))
            if re.match(r"(\d+)", x)
            else float("inf")
        )
    )

    # Create the table of contents structure
    toc = {
        "format": "jb-book",
        "root": "index",
        "chapters": [
            {
                "file": os.path.join(relative_path, "index"),
                "title": path_parts[-1].replace("-", " ").title(),
                "sections": [
                    {"file": os.path.join(relative_path, os.path.splitext(f)[0])}
                    for f in files
                    if f != "index.md"
                ],
            }
        ],
    }

    # Save _toc.yml in the specified folder
    toc_file_path = os.path.join(folder_path, "_toc.yml")
    with open(toc_file_path, "w", encoding="utf-8") as f:
        yaml.dump(toc, f, default_flow_style=False, sort_keys=False)

    print(f"_toc.yml has been generated and saved to {toc_file_path}")


def main():
    """
    Main function to parse arguments and generate the TOC.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate _toc.yml for a JupyterBook folder."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder where _toc.yml will be generated.",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Generate TOC
        generate_toc(args.folder_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
