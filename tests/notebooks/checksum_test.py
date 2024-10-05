import pytest
from unittest.mock import mock_open, patch
import json
import hashlib
from m3util.notebooks.checksum import calculate_notebook_checksum


def test_calculate_notebook_checksum_sha256():
    # Sample notebook content with code and markdown cells
    notebook_content = {
        "cells": [
            {"cell_type": "code", "source": ["print('Hello World')\n"]},
            {"cell_type": "markdown", "source": ["# This is a markdown cell"]},
            {"cell_type": "code", "source": ["a = 10\n", "b = 20\n", "print(a + b)\n"]},
        ]
    }

    # Expected concatenated code cells content
    code_cells_content = (
        "print('Hello World')\n"
        "\n"
        "a = 10\nb = 20\nprint(a + b)\n"
        "\n"
    )

    # Compute expected checksum using SHA-256
    expected_checksum = hashlib.sha256(code_cells_content.encode('utf-8')).hexdigest()

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function
        checksum = calculate_notebook_checksum('dummy_path.ipynb', algorithm='sha256')

        # Assert the checksum matches the expected value
        assert checksum == expected_checksum


def test_calculate_notebook_checksum_md5():
    # Same notebook content as before
    notebook_content = {
        "cells": [
            {"cell_type": "code", "source": ["print('Hello World')\n"]},
            {"cell_type": "markdown", "source": ["# This is a markdown cell"]},
            {"cell_type": "code", "source": ["a = 10\n", "b = 20\n", "print(a + b)\n"]},
        ]
    }

    # Expected concatenated code cells content
    code_cells_content = (
        "print('Hello World')\n"
        "\n"
        "a = 10\nb = 20\nprint(a + b)\n"
        "\n"
    )

    # Compute expected checksum using MD5
    expected_checksum = hashlib.md5(code_cells_content.encode('utf-8')).hexdigest()

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function with MD5 algorithm
        checksum = calculate_notebook_checksum('dummy_path.ipynb', algorithm='md5')

        # Assert the checksum matches the expected value
        assert checksum == expected_checksum


def test_calculate_notebook_checksum_sha1():
    # Same notebook content as before
    notebook_content = {
        "cells": [
            {"cell_type": "code", "source": ["print('Hello World')\n"]},
            {"cell_type": "markdown", "source": ["# This is a markdown cell"]},
            {"cell_type": "code", "source": ["a = 10\n", "b = 20\n", "print(a + b)\n"]},
        ]
    }

    # Expected concatenated code cells content
    code_cells_content = (
        "print('Hello World')\n"
        "\n"
        "a = 10\nb = 20\nprint(a + b)\n"
        "\n"
    )

    # Compute expected checksum using SHA-1
    expected_checksum = hashlib.sha1(code_cells_content.encode('utf-8')).hexdigest()

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function with SHA-1 algorithm
        checksum = calculate_notebook_checksum('dummy_path.ipynb', algorithm='sha1')

        # Assert the checksum matches the expected value
        assert checksum == expected_checksum


def test_calculate_notebook_checksum_no_code_cells():
    # Notebook content with only markdown cells
    notebook_content = {
        "cells": [
            {"cell_type": "markdown", "source": ["# This is a markdown cell"]},
            {"cell_type": "markdown", "source": ["## Another markdown cell"]},
        ]
    }

    # Expected concatenated code cells content is empty
    code_cells_content = ""

    # Compute expected checksum using SHA-256
    expected_checksum = hashlib.sha256(code_cells_content.encode('utf-8')).hexdigest()

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function
        checksum = calculate_notebook_checksum('dummy_path.ipynb')

        # Assert the checksum matches the expected value
        assert checksum == expected_checksum


def test_calculate_notebook_checksum_empty_notebook():
    # Empty notebook content
    notebook_content = {"cells": []}

    # Expected concatenated code cells content is empty
    code_cells_content = ""

    # Compute expected checksum using SHA-256
    expected_checksum = hashlib.sha256(code_cells_content.encode('utf-8')).hexdigest()

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function
        checksum = calculate_notebook_checksum('dummy_path.ipynb')

        # Assert the checksum matches the expected value
        assert checksum == expected_checksum


def test_calculate_notebook_checksum_invalid_algorithm():
    # Notebook content (can be empty since it should not reach hashing)
    notebook_content = {"cells": []}

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function with an invalid algorithm and expect a ValueError
        with pytest.raises(ValueError):
            calculate_notebook_checksum('dummy_path.ipynb', algorithm='invalid_algo')


def test_calculate_notebook_checksum_invalid_json():
    # Invalid JSON content
    invalid_json_content = "this is not valid JSON"

    # Mock open to return invalid JSON content
    with patch('builtins.open', mock_open(read_data=invalid_json_content)):
        # Call the function and expect a JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            calculate_notebook_checksum('dummy_path.ipynb')


def test_calculate_notebook_checksum_missing_cells_key():
    # Notebook content without 'cells' key
    notebook_content = {
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Mock open and json.load to return the notebook content
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function and expect a KeyError
        with pytest.raises(KeyError):
            calculate_notebook_checksum('dummy_path.ipynb')


def test_calculate_notebook_checksum_empty_source():
    # Notebook with a code cell that has an empty source
    notebook_content = {
        "cells": [
            {"cell_type": "code", "source": []},
            {"cell_type": "code", "source": ["\n"]},
        ]
    }

    # Expected concatenated code cells content
    code_cells_content = "\n\n\n"

    # Compute expected checksum using SHA-256
    expected_checksum = hashlib.sha256(code_cells_content.encode('utf-8')).hexdigest()

    # Mock open and json.load
    with patch('builtins.open', mock_open(read_data=json.dumps(notebook_content))):
        # Call the function
        checksum = calculate_notebook_checksum('dummy_path.ipynb')

        # Assert the checksum matches the expected value
        assert checksum == expected_checksum
