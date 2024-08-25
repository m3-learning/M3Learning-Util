import pytest
import h5py
import numpy as np
import os

from m3util.util.h5 import (
    print_tree,
    get_tree,
    make_group,
    make_dataset,
    find_groups_with_string,
    find_measurement,
)


@pytest.fixture
def h5_file(tmp_path):
    """Fixture to create a temporary HDF5 file."""
    file_path = tmp_path / "test.h5"
    with h5py.File(file_path, "w") as f:
        grp1 = f.create_group("group1")
        grp1.create_dataset("dataset1", data=np.arange(10))
        grp2 = f.create_group("group2")
        grp2.create_dataset("dataset2", data=np.arange(10, 20))
        grp3 = grp1.create_group("nested_group")
        grp3.create_dataset("dataset3", data=np.arange(20, 30))
    yield file_path


def test_print_tree(h5_file, capsys):
    """Test the print_tree function."""
    with h5py.File(h5_file, "r") as f:
        print_tree(f)
    
    captured = capsys.readouterr()
    output = captured.out.splitlines()
    expected_output = ["/", "/group1", "/group1/dataset1", "/group1/nested_group", "/group1/nested_group/dataset3", "/group2", "/group2/dataset2"]
    assert output == expected_output


def test_get_tree(h5_file):
    """Test the get_tree function."""
    with h5py.File(h5_file, "r") as f:
        tree = get_tree(f)
    
    expected_tree = ["/", "/group1", "/group1/dataset1", "/group1/nested_group", "/group1/nested_group/dataset3", "/group2", "/group2/dataset2"]
    assert tree == expected_tree


def test_make_group(h5_file):
    """Test the make_group function."""
    with h5py.File(h5_file, "a") as f:
        new_group = make_group(f, "group3")
        assert "group3" in f
        assert isinstance(new_group, h5py.Group)


def test_make_group_existing(h5_file, capsys):
    """Test the make_group function with an existing group."""
    with h5py.File(h5_file, "a") as f:
        make_group(f, "group1")
    
    captured = capsys.readouterr()
    assert "Could not add group - it might already exist." in captured.out


def test_make_dataset(h5_file):
    """Test the make_dataset function."""
    data = np.arange(30, 40)
    with h5py.File(h5_file, "a") as f:
        make_dataset(f["group1"], "dataset4", data)
        assert "dataset4" in f["group1"]
        np.testing.assert_array_equal(f["group1/dataset4"][:], data)


def test_make_dataset_overwrite(h5_file):
    """Test the make_dataset function by overwriting an existing dataset."""
    data = np.arange(30, 40)
    with h5py.File(h5_file, "a") as f:
        make_dataset(f["group1"], "dataset1", data)
        np.testing.assert_array_equal(f["group1/dataset1"][:], data)


def test_find_groups_with_string(h5_file):
    """Test the find_groups_with_string function."""
    result = find_groups_with_string(h5_file, "nested")
    # Normalize the expected result to match the format returned by the function
    expected_result = ["/group1/nested_group"]
    assert result == expected_result


def test_find_groups_with_string_not_found(h5_file):
    """Test the find_groups_with_string function with a string that doesn't match any group."""
    result = find_groups_with_string(h5_file, "nonexistent")
    assert result == []


def test_find_measurement(h5_file):
    """Test the find_measurement function."""
    result = find_measurement(h5_file, "dataset3", "/group1/nested_group")
    assert result == "dataset3"

def test_find_measurement_not_found(h5_file):
    """Test the find_measurement function with a string that doesn't match any dataset."""
    result = find_measurement(h5_file, "nonexistent", "/group1")
    assert result == []

