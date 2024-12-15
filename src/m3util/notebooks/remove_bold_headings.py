import re
import sys


def remove_bold_from_headings(input_file: str):
    """
    Reads a Markdown file, removes bold text within headings, and overwrites the file.

    Args:
        input_file (str): Path to the input Markdown file.
    """
    # Read the Markdown file
    with open(input_file, "r", encoding="utf-8") as file:
        markdown_content = file.readlines()

    # Regex pattern to match headings with bold text
    heading_pattern = re.compile(r"^(#{1,6}\s.*?)(\*\*.*?\*\*|__.*?__)(.*)$")

    modified_content = []

    for line in markdown_content:
        if line.strip().startswith("#"):  # Check if the line is a heading
            line = heading_pattern.sub(r"\1\3", line)
        modified_content.append(line)

    # Overwrite the input file with modified content
    with open(input_file, "w", encoding="utf-8") as file:
        file.writelines(modified_content)

    print(f"Modified Markdown saved to: {input_file}")


def main(args=None):
    """
    Entry point for the script.
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: remove-bold-headings <input_file>")
        sys.exit(1)

    input_file = args[0]

    try:
        remove_bold_from_headings(input_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
