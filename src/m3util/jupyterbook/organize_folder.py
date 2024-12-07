import os
import shutil
import re
import argparse
import sys


def organize_folder(path="."):
    try:
        # Normalize the path
        path = os.path.abspath(path)

        # Check if the provided path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path '{path}' does not exist.")

        # Create assets/figures directory within the provided path
        assets_dir = os.path.join(path, "assets", "figures")
        os.makedirs(assets_dir, exist_ok=True)

        # Get all files in the provided directory (excluding subdirectories)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Move image files to the assets/figures directory
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                src = os.path.join(path, file)
                dst = os.path.join(assets_dir, file)
                if os.path.abspath(src) != os.path.abspath(
                    dst
                ):  # Avoid moving files already in target
                    shutil.move(src, dst)
                    print(f"Moved: {file} -> {dst}")

        # Refresh the list of files in the provided path after moving images
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Filter for .ipynb and .md files, excluding index.md
        files = [
            f
            for f in files
            if f.endswith(".ipynb") or f.endswith(".md") and f != "index.md"
        ]

        # Process files for renaming with no gaps in numbering
        number_pattern = re.compile(r"^(\d+)(?:[-_](\d+))?(.*)$")
        files_with_numbers = []

        for file in files:
            match = number_pattern.match(file)
            if match:
                main_num = int(match.group(1))
                sub_num = int(match.group(2)) if match.group(2) else 0
                remainder = match.group(3)
                # Standardize the separator to `_`
                remainder = re.sub(r"^[-_]", "_", remainder)
                files_with_numbers.append((file, main_num, sub_num, remainder))
            else:
                # Assign a default numeric prefix for files without numbers
                files_with_numbers.append((file, float("inf"), 0, file))

        # Sort files by main number and sub number
        files_with_numbers.sort(key=lambda x: (x[1], x[2]))

        # Assign new sequential numbers starting from 1
        rename_map = {}
        counter = 1
        for file, _, _, remainder in files_with_numbers:
            new_name = f"{counter}{remainder}"
            rename_map[file] = new_name
            counter += 1

        # Perform renaming within the provided path
        for old_name, new_name in rename_map.items():
            src = os.path.join(path, old_name)
            dst = os.path.join(path, new_name)
            shutil.move(src, dst)
            print(f"Renamed: {old_name} -> {new_name}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Organize a folder by moving images and renaming files with no gaps in numbering."
    )
    parser.add_argument("path", type=str, help="The path to the folder to organize")
    args = parser.parse_args()
    organize_folder(args.path)


if __name__ == "__main__":
    main()
