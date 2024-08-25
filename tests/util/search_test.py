import pytest
import re
import fnmatch
import os
from m3util.util.search import (
    in_list,
    get_tuple_names,
    extract_number,
    save_list_to_txt,
)  # replace `your_module_name` with the actual module name


def test_in_list():
    """Test the in_list function."""
    list_ = ["apple", "banana", "cherry", "date"]
    pattern = "b*"
    assert in_list(list_, pattern) is True, "Pattern should match items in the list"

    pattern = "z*"
    assert (
        in_list(list_, pattern) is False
    ), "Pattern should not match any items in the list"


def test_get_tuple_names():
    """Test the get_tuple_names function."""
    a = 10
    b = 20
    c = 30
    data = (a, b, c)

    result = get_tuple_names(data, locals())
    expected = ["a", "b", "c"]

    assert set(result) == set(expected), f"Expected {expected} but got {result}"


def test_extract_number():
    """Test the extract_number function."""
    assert extract_number("abc123") == 123, "Should extract integer from string"
    assert (
        extract_number("price is 45.67 dollars") == 45.67
    ), "Should extract float from string"
    assert (
        extract_number("no numbers here") is None
    ), "Should return None when no numbers are present"
    assert (
        extract_number("multiple 123 and 456") == 123
    ), "Should extract the first number found"


def test_save_list_to_txt(tmp_path):
    """Test the save_list_to_txt function."""
    lst = ["item1", "item2", "item3"]
    filename = tmp_path / "test_file.txt"

    save_list_to_txt(lst, filename)

    with open(filename, "r") as file:
        lines = file.readlines()

    expected_lines = ["item1\n", "item2\n", "item3\n"]
    assert lines == expected_lines, f"Expected {expected_lines} but got {lines}"
