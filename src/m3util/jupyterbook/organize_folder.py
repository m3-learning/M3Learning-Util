import os
import shutil
import re


def organize_folder(path="."):
    # Create assets/figures directory if it doesn't exist
    assets_dir = os.path.join(path, "assets", "figures")
    os.makedirs(assets_dir, exist_ok=True)

    # Get all files in directory
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # Move image files
    image_extensions = {".png", ".jpg", ".jpeg", ".gif"}
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            src = os.path.join(path, file)
            dst = os.path.join(assets_dir, file)
            shutil.move(src, dst)

    # Process files for renaming
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
            # Calculate effective number
            effective_num = main_num if sub_num == 0 else main_num + sub_num
            files_with_numbers.append((file, effective_num, remainder))

    # Sort files by calculated effective number
    files_with_numbers.sort(key=lambda x: x[1])

    # Ensure unique numbering and map to new names
    used_numbers = set()
    rename_map = {}

    for file, effective_num, remainder in files_with_numbers:
        # Increment effective_num if already used
        while effective_num in used_numbers:
            effective_num += 1
        used_numbers.add(effective_num)

        new_name = f"{effective_num}{remainder}"
        rename_map[file] = new_name

    # Perform renaming
    for old_name, new_name in rename_map.items():
        src = os.path.join(path, old_name)
        dst = os.path.join(path, new_name)
        shutil.move(src, dst)


if __name__ == "__main__":
    organize_folder()
