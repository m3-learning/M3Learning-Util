from m3util.util.generate_init import generate_init_py
import os
import pytest

def test_generate_init_py(tmp_path):
    # Create a mock package structure
    package_dir = tmp_path / "my_package"
    package_dir.mkdir()
    
    # Create some Python files in the package directory
    (package_dir / "module1.py").write_text("# Module 1")
    (package_dir / "module2.py").write_text("# Module 2")

    # Create a subpackage with its own __init__.py
    subpackage_dir = package_dir / "subpackage"
    subpackage_dir.mkdir()
    (subpackage_dir / "__init__.py").write_text("# Subpackage init")
    (subpackage_dir / "module3.py").write_text("# Module 3")

    # Create another directory that should be ignored (__pycache__)
    pycache_dir = package_dir / "__pycache__"
    pycache_dir.mkdir()

    # Run the generate_init_py function
    generate_init_py(str(package_dir))

    # Check the contents of the generated __init__.py file in the package_dir
    init_file_content = (package_dir / "__init__.py").read_text()

    expected_init_content = (
        "# Auto-generated __init__.py\n\n"
        "from . import module1\n"
        "from . import module2\n"
        "from . import subpackage\n\n"
        "__all__ = ['module1', 'module2', 'subpackage']\n"
    )

    for line in expected_init_content:
        assert line in init_file_content

    # Check the contents of the regenerated __init__.py file in the subpackage_dir
    subpackage_init_content = (subpackage_dir / "__init__.py").read_text()

    expected_subpackage_init_content = (
        "# Auto-generated __init__.py\n\n"
        "from . import module3\n\n"
        "__all__ = ['module3']\n"
    )
    
    for line in subpackage_init_content:
        assert line in expected_subpackage_init_content

    # Check that __pycache__ was ignored and no __init__.py was generated there
    assert not (pycache_dir / "__init__.py").exists()
