import os
import yaml
import re


def generate_toc(path="."):
    # Get absolute path and split into components
    abs_path = os.path.abspath(path)
    path_parts = abs_path.split(os.sep)

    # Find index of 'jupyterbook' in path
    try:
        jb_index = path_parts.index("jupyterbook")
        relative_path = os.sep.join(path_parts[jb_index + 1 :])
    except ValueError:
        relative_path = path_parts[-1]

    # Get files and sort
    files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
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

    with open("_toc.yml", "w", encoding="utf-8") as f:
        yaml.dump(toc, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    generate_toc()
