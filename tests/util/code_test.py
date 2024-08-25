import pytest
import inspect
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from m3util.util.code import print_code


def test_print_code(capsys):
    # Define a sample function to test
    def sample_function():
        x = 1
        y = 2
        return x + y

    # Call the print_code function with the sample function
    print_code(sample_function)

    # Capture the output
    captured = capsys.readouterr()

    # Fetch source lines and starting line number
    source_lines, starting_line_number = inspect.getsourcelines(sample_function)
    source_code = "".join(source_lines)

    # Perform syntax highlighting using Pygments
    expected_highlighted_code = highlight(
        source_code, PythonLexer(), TerminalFormatter())

    # Rebuild the expected output with line numbers
    expected_output = "\n".join(
        f"{i}: {line}" for i, line in enumerate(expected_highlighted_code.split("\n"), start=starting_line_number)
    )

    # Compare the captured output to the expected output
    assert captured.out.strip() == expected_output.strip()

