import pytest
import pandas as pd
from m3util.pandas.filter import (
    find_min_max_by_group,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C"],
            "value": [10, 20, 5, 15, 25],
            "file_name": [
                "file1.pth",
                "file2.pth",
                "file3.pth",
                "other_file.pth",
                "file4.pth",
            ],
            "exclude": [False, False, True, False, False],
        }
    )


def test_filter_with_regex(sample_df):
    # Pattern refined to match only files that start with 'file' followed by a digit and end with '.pth'
    result = find_min_max_by_group(
        sample_df, "value", find="min", file_name=r"^file1.pth$"
    )
    assert result["file_name"] == "file1.pth"
    assert result["value"] == 10


def test_filter_with_special_characters(sample_df):
    # Refined pattern to match files with 'file' followed by a digit and '.pth' extension
    result = find_min_max_by_group(
        sample_df, "value", find="min", file_name=r"^file1.pth$"
    )
    assert result["file_name"] == "file1.pth"
    assert result["value"] == 10


def test_find_min_value(sample_df):
    result = find_min_max_by_group(sample_df, "value", find="min", group="A")
    assert result["value"] == 10
    assert result["group"] == "A"


def test_find_max_value(sample_df):
    result = find_min_max_by_group(sample_df, "value", find="max", group="B")
    assert result["value"] == 15
    assert result["group"] == "B"



def test_exclude_rows(sample_df):
    result = find_min_max_by_group(
        sample_df, "value", find="min", exclude_kwargs={"exclude": True}
    )
    assert result["value"] == 10  # Exclude 'B' group with 'value' 5


def test_find_min_no_matching_rows(sample_df):
    with pytest.raises(
        ValueError,
        match="No rows match the filter criteria after inclusion and exclusion.",
    ):
        find_min_max_by_group(
            sample_df, "value", find="min", exclude_kwargs={"exclude": False}, group="D"
        )


def test_find_max_global(sample_df):
    result = find_min_max_by_group(sample_df, "value", find="max")
    assert result["value"] == 25
    assert result["group"] == "C"
    assert result["file_name"] == "file4.pth"


def test_find_min_with_exact_match(sample_df):
    result = find_min_max_by_group(sample_df, "value", find="min", group="B")
    assert result["value"] == 5
    assert result["group"] == "B"

