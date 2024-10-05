from m3util.util.generate_init import generate_init_py, main
import sys
from unittest.mock import patch
from unittest.mock import mock_open, call

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

# Helper function to simulate command-line argument input
def mock_argv(*args):
    sys.argv = list(args)


@patch("sys.exit")
@patch("builtins.print")
@patch("os.path.isdir")
def test_main_invalid_directory(mock_isdir, mock_print, mock_exit):
    """Test when an invalid directory is provided."""
    # Simulate running the script with an invalid directory argument
    mock_argv("generate_init_py", "invalid_directory")

    # Mock os.path.isdir to return False for the invalid directory
    mock_isdir.return_value = False

    # Call the main function
    main()

    # Check that the error message was printed
    mock_print.assert_any_call("Error: invalid_directory is not a valid directory")

    # Check that sys.exit(1) was called
    mock_exit.assert_called_once_with(1)


@patch("builtins.print")
@patch("os.path.isdir")
@patch("m3util.util.generate_init.generate_init_py")
def test_main_valid_directory(mock_generate_init_py, mock_isdir, mock_print):
    """Test when a valid directory is provided and __init__.py is successfully generated."""
    # Simulate running the script with a valid directory argument
    mock_argv("generate_init_py", "valid_directory")

    # Mock os.path.isdir to return True for the valid directory
    mock_isdir.return_value = True

    # Call the main function
    main()

    # Check that generate_init_py was called with the correct directory
    mock_generate_init_py.assert_called_once_with("valid_directory")

    # Check that the success message was printed
    mock_print.assert_called_once_with(
        "__init__.py generated in valid_directory and its subdirectories, excluding __pycache__ directories"
    )


@patch("os.path.isfile")
@patch("os.walk")
@patch("builtins.open", new_callable=mock_open)
def test_generate_init_py_(mock_open_func, mock_os_walk, mock_isfile):
    """Test generate_init_py with a valid directory and Python files."""
    # Mock the os.walk to simulate directory traversal
    mock_os_walk.return_value = [
        ("root", ["subdir"], ["file1.py", "file2.py", "__init__.py"]),
        ("root/subdir", [], ["subfile.py", "__init__.py"]),
    ]

    # Mock os.path.isfile to return True for subpackage __init__.py
    mock_isfile.side_effect = lambda path: "__init__.py" in path

    # Call the generate_init_py function
    generate_init_py("root")

    # Check that open was called for each directory
    calls = [
        call("root/__init__.py", "w"),
        call("root/subdir/__init__.py", "w"),
    ]
    mock_open_func.assert_has_calls(calls, any_order=True)

    # Check the content written to the __init__.py files
    handle = mock_open_func()
    handle.write.assert_any_call("# Auto-generated __init__.py\n\n")
    handle.write.assert_any_call("from . import file1\n")
    handle.write.assert_any_call("from . import file2\n")
    handle.write.assert_any_call("from . import subfile\n")



@patch("os.path.isfile")
@patch("os.walk")
@patch("builtins.open", new_callable=mock_open)
def test_generate_init_py_with_no_python_files(
    mock_open_func, mock_os_walk, mock_isfile
):
    """Test generate_init_py with directories that contain no Python files."""
    # Mock the os.walk to simulate directory traversal with no Python files
    mock_os_walk.return_value = [
        ("root", ["subdir"], ["not_python.txt", "__init__.py"]),
        ("root/subdir", [], ["data.csv"]),
    ]

    # Mock os.path.isfile to return True for subpackage __init__.py
    mock_isfile.side_effect = lambda path: "__init__.py" in path

    # Call the generate_init_py function
    generate_init_py("root")

    # Check that open was called for each directory
    calls = [
        call("root/__init__.py", "w"),
        call("root/subdir/__init__.py", "w"),
    ]
    mock_open_func.assert_has_calls(calls, any_order=True)

    # Check that no Python file imports were written
    handle = mock_open_func()
    handle.write.assert_any_call("# Auto-generated __init__.py\n\n")
    handle.write.assert_any_call("\n__all__ = []\n")
