import pandas as pd
import re


def find_min_max_by_group(
    df, optimized_result, find="min", exclude_kwargs=None, **kwargs
):
    """
    Finds the minimum or maximum value for a specific column based on filtered criteria,
    and returns the corresponding row from the original DataFrame.

    Parameters:
    df (pd.DataFrame): The original DataFrame to filter, group, and search.
    optimized_result (str): The column name to find the max or min value.
    find (str): 'min' or 'max' to specify whether to find the minimum or maximum.
    exclude_kwargs (dict): Column-value pairs to exclude from the DataFrame.
    **kwargs: Column-value pairs to filter the DataFrame.

    Returns:
    pd.Series: A Series containing the min/max row data from the original DataFrame.
    """

    # Apply filtering criteria to create a filtered DataFrame
    filtered_df = filter_df(df, **kwargs)

    # Exclude rows based on the exclusion keyword arguments
    if exclude_kwargs:
        for ex_key, ex_value in exclude_kwargs.items():
            filtered_df = filtered_df[filtered_df[ex_key] != ex_value]

    # Check if any rows match the filters
    if filtered_df.empty:
        raise ValueError(
            "No rows match the filter criteria after inclusion and exclusion."
        )

    # Find the index of the min/max in the filtered DataFrame
    if find == "min":
        idx = filtered_df[optimized_result].idxmin()
    elif find == "max":
        idx = filtered_df[optimized_result].idxmax()
    else:
        raise ValueError("Invalid value for 'find'. Use 'min' or 'max'.")

    # Return the corresponding row from the original DataFrame
    return df.loc[idx]


def filter_df(df, **kwargs):
    for key, value in kwargs.items():
        if pd.api.types.is_numeric_dtype(df[key]):
            # Use exact match for numeric columns
            df = df[df[key] == value]
        else:
            # Determine if `value` contains regex characters
            if re.search(r"[*+?^]", str(value)):
                # Treat as regex if it contains special characters
                try:
                    df = df[
                        df[key].astype(str).str.contains(value, regex=True, na=False)
                    ]
                except re.error as e:
                    print(f"Regex error on column '{key}' with pattern '{value}': {e}")
                    raise
            else:
                # Treat as literal if no special characters found
                pattern = str(value)
                df = df[
                    df[key].astype(str).str.contains(pattern, regex=False, na=False)
                ]
    return df
