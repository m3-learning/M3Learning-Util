def find_min_max_by_group(df, col_name, find="min", exclude_kwargs=None, **kwargs):
    """
    Finds the minimum or maximum value for a specific column, grouped by unique values
    of the specified kwargs, while excluding rows based on exclude_kwargs.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter, group, and search.
    col_name (str): The column name to find the max or min value.
    find (str): 'min' or 'max' to specify whether to find the minimum or maximum.
    exclude_kwargs (dict): Column-value pairs to exclude from the DataFrame.
    **kwargs: Column-value pairs to group by (values of these columns will be grouped).

    Returns:
    pd.DataFrame: A DataFrame containing the min/max row from each group.
    """

    # Filter the DataFrame based on the keyword arguments for inclusion
    for key, value in kwargs.items():
        df = df[df[key] == value]

    # Exclude rows based on the exclusion keyword arguments
    if exclude_kwargs:
        for ex_key, ex_value in exclude_kwargs.items():
            df = df[df[ex_key] != ex_value]

    # Check if any rows match the filters
    if df.empty:
        raise ValueError(
            "No rows match the filter criteria after inclusion and exclusion."
        )

    # Group the DataFrame by the kwargs' columns
    group_cols = list(kwargs.keys())
    grouped_df = df.groupby(group_cols)

    # Select only the columns we need for aggregation (excluding the grouping columns)
    df_to_agg = df.drop(columns=group_cols)

    # Define a function to return the index of the min/max for each group
    def get_min_max_index(group):
        if find == "min":
            return group[col_name].idxmin()
        elif find == "max":
            return group[col_name].idxmax()
        else:
            raise ValueError("Invalid value for 'find'. Use 'min' or 'max'.")

    # Apply the function to each group and get the index of the min/max row
    min_max_indices = grouped_df[df_to_agg.columns].apply(get_min_max_index)

    # Extract the corresponding rows from the original dataframe
    return df.loc[min_max_indices]
