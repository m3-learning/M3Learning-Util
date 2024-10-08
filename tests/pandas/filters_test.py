import pandas as pd
import pytest
from m3util.pandas.filter import find_min_max_by_group, filter_df


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


# Sample DataFrame for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "column1": ["apple", "banana", "cherry", "date"],
            "column2": ["fruit", "berry", "fruit", "dry fruit"],
            "column3": [10, 20, 30, 40],  # Integer column
            "column4": [1.5, 2.5, 3.5, 4.5],  # Float column
        }
    )


def test_regex_match_string(sample_df):
    result = filter_df(sample_df, column1="a.*")
    # Adjusted expectation to match rows with 'a' anywhere in 'column1'
    assert len(result) == 3  # 'apple', 'banana', and 'date' contain 'a'
    assert set(result["column1"]) == {"apple", "banana", "date"}


def test_multiple_conditions(sample_df):
    result = filter_df(sample_df, column1="a.*", column2="fruit")
    # Adjusted expectation since 'apple' and 'date' match 'a.*' and contain 'fruit' in 'column2'
    assert len(result) == 2
    assert set(result["column1"]) == {"apple", "date"}

def test_exact_match_int(sample_df):
    result = filter_df(sample_df, column3=10)
    assert len(result) == 1
    assert result.iloc[0]["column1"] == "apple"


def test_exact_match_float(sample_df):
    result = filter_df(sample_df, column4=2.5)
    assert len(result) == 1
    assert result.iloc[0]["column1"] == "banana"


def test_mixed_type_filtering(sample_df):
    result = filter_df(sample_df, column1="a.*", column3=10)
    assert len(result) == 1
    assert result.iloc[0]["column1"] == "apple"


def test_no_match(sample_df):
    result = filter_df(sample_df, column1="nonexistent")
    assert len(result) == 0