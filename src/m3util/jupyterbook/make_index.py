import os
import argparse


def generate_index(folder_path, output_file, title, start_number):
    """
    Generate an index.md file for a JupyterBook with a single session.

    Args:
        folder_path (str): The path to the folder containing the markdown and notebook files.
        output_file (str): The name of the output index file.
        title (str): Custom title for the session.
        start_number (int): Starting number for the file entries.
    """
    # Get a list of .md and .ipynb files in the folder
    files = [
        f for f in os.listdir(folder_path) if f.endswith(".md") or f.endswith(".ipynb")
    ]

    # Sort files alphanumerically
    files.sort()

    # Generate index content
    content = [f"# {title}"]
    for i, file in enumerate(files, start=start_number):
        doc_title = (
            file.split("_", 2)[-1]
            .replace(".md", "")
            .replace(".ipynb", "")
            .replace("_", " ")
        )
        content.append(f"{i}. {{doc}}`{doc_title} <./{file}>`")

    # Write to the index.md file
    with open(output_file, "w") as f:
        f.write("\n".join(content))

    print(f"Index file '{output_file}' created successfully.")


if __name__ == "__main__":
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
