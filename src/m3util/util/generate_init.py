import os
import sys


def generate_init_py(package_dir):
    """
    Auto-generates __init__.py files in the package directory and all subdirectories,
    excluding __pycache__ directories. Imports all modules and subpackages, adding them
    to the __all__ list.
    """
    for root, dirs, files in os.walk(package_dir):
        # Exclude __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]

        modules = []
        init_file_path = os.path.join(root, "__init__.py")
        with open(init_file_path, "w") as init_file:
            init_file.write("# Auto-generated __init__.py\n\n")

            # Add import statements for Python files (modules)
            for filename in files:
                if filename.endswith(".py") and filename != "__init__.py":
                    module_name = filename[:-3]
                    modules.append(module_name)
                    init_file.write(f"from . import {module_name}\n")

            # Add import statements for subdirectories (subpackages)
            for subdir in dirs:
                if os.path.isfile(os.path.join(root, subdir, "__init__.py")):
                    modules.append(subdir)
                    init_file.write(f"from . import {subdir}\n")

            init_file.write(f"\n__all__ = {modules}\n")

def main():
            
    if len(sys.argv) != 2:
        print("Usage: python generate_init_py <package_directory>")
        sys.exit(1)

    package_dir = sys.argv[1]
    if not os.path.isdir(package_dir):
        print(f"Error: {package_dir} is not a valid directory")
        sys.exit(1)

    generate_init_py(package_dir)
    print(
        f"__init__.py generated in {package_dir} and its subdirectories, excluding __pycache__ directories"
    )


if __name__ == "__main__":
    main()