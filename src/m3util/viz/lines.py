def handle_linewidth_conflicts(textprops):
    """
    Handle both 'linewidth' and 'lw' in the text properties, selecting the larger value if both are provided.
    Also forces 'linestyle' to be 'solid'.

    Args:
        textprops (dict): A dictionary of text properties.

    Returns:
        dict: The updated text properties with conflicts resolved.
    """
    # Handle both 'linewidth' and 'lw', selecting the larger value if both are provided
    if "lw" in textprops and "linewidth" in textprops:
        # Select the larger of the two
        larger_linewidth = max(textprops["lw"], textprops["linewidth"])
        # Remove the smaller one to avoid conflicts
        textprops.pop("lw")
        textprops["linewidth"] = larger_linewidth
    elif "lw" in textprops:
        # If only 'lw' is present, map it to 'linewidth'
        textprops["linewidth"] = textprops.pop("lw")

    if "ls" in textprops and "linestyle" in textprops:
        textprops.pop("ls")
        textprops["linestyle"] = "solid"

    return textprops


def draw_lines(ax, x_values, y_values, style=None, halo=None):
    """
    Draws lines on a matplotlib axis based on x and y values, with an optional halo effect.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the lines.
        x_values (list or array): A list or array of x-values for the line.
        y_values (list or array): A list or array of y-values for the line.
        style (dict, optional): A dictionary specifying line style properties.
                                Keys can include 'color', 'linewidth', 'linestyle', etc.
        halo (dict, optional): A dictionary specifying halo properties.
                               Keys can include 'enabled', 'color', 'scale'.
                               Defaults to None (no halo).

    Returns:
        matplotlib.lines.Line2D: The line artist object.
    """
    # Default style if none is provided
    if style is None:
        style = {}

    try:
        line = ax.plot(x_values, y_values, **style)
    except TypeError as e:
        raise TypeError(f"Invalid style parameter: {e}")

    if ax is None:
        raise TypeError("The axis object cannot be None")

    if not x_values or not y_values:
        raise ValueError("x_values and y_values cannot be empty")

    # Handle linewidth conflicts in the main line style
    style = handle_linewidth_conflicts(style)

    # Default halo settings if not provided
    halo_defaults = {
        "enabled": False,  # Whether to enable the halo or not
        "color": "white",  # Default halo color
        "scale": 3,  # Scale the halo size relative to the linewidth
    }
    halo_settings = {**halo_defaults, **(halo or {})}

    # If a halo is enabled, draw a thick line beneath the main line
    if halo_settings["enabled"]:
        halo_style = style.copy()  # Copy the main style for the halo
        halo_linewidth = (
            style.get("linewidth", style.get("lw", 2)) * halo_settings["scale"]
        )
        halo_style["linewidth"] = halo_linewidth
        halo_style["lw"] = halo_linewidth
        halo_style["color"] = halo_settings["color"]  # Halo color
        halo_style["linestyle"] = (
            "solid"  # Halo should always be solid, regardless of main line style
        )
        halo_style = handle_linewidth_conflicts(
            halo_style
        )  # Handle conflicts between 'linewidth' and 'lw'

        print(halo_style)
        ax.plot(x_values, y_values, **halo_style)

    # Draw the actual line with the provided style
    line = ax.plot(x_values, y_values, **style)

    return line
