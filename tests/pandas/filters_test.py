import pandas as pd
import pytest
from m3util.pandas.filter import find_min_max_by_group


def test_find_min_max_by_group_min():
    data = {"A": ["foo", "foo", "bar", "bar"], "B": [1, 2, 3, 4], "C": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    result = find_min_max_by_group(df, "C", find="min", A="foo")
    expected = df.iloc[[0]]
    pd.testing.assert_frame_equal(result, expected)


def test_find_min_max_by_group_max():
    data = {"A": ["foo", "foo", "bar", "bar"], "B": [1, 2, 3, 4], "C": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    result = find_min_max_by_group(df, "C", find="max", A="bar")
    expected = df.iloc[[3]]
    pd.testing.assert_frame_equal(result, expected)


def test_find_min_max_by_group_exclude():
    data = {"A": ["foo", "foo", "bar", "bar"], "B": [1, 2, 3, 4], "C": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    result = find_min_max_by_group(
        df, "C", find="max", exclude_kwargs={"B": 4}, A="bar"
    )
    expected = df.iloc[[2]]
    pd.testing.assert_frame_equal(result, expected)


def test_find_min_max_by_group_no_match():
    data = {"A": ["foo", "foo", "bar", "bar"], "B": [1, 2, 3, 4], "C": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    with pytest.raises(
        ValueError,
        match="No rows match the filter criteria after inclusion and exclusion.",
    ):
        find_min_max_by_group(df, "C", find="min", A="baz")


def test_find_min_max_by_group_invalid_find():
    data = {"A": ["foo", "foo", "bar", "bar"], "B": [1, 2, 3, 4], "C": [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    with pytest.raises(
        ValueError, match="Invalid value for 'find'. Use 'min' or 'max'."
    ):
        find_min_max_by_group(df, "C", find="average", A="foo")
