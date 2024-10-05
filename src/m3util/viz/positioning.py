import matplotlib.pyplot as plt


def obj_offset(xy, offset=None, offset_units="fontsize", ax=None, **kwargs):
    """
    Offset a point on the plot based on the specified units.

    Parameters:
        xy (tuple): The (x, y) position of the object.
        offset (tuple or str): Offset values in (offset_x, offset_y) or direction ('left', 'right', 'up', 'down').
        offset_units (str): The units for the offset ('fontsize', 'inches', 'points', 'fraction'). Default is 'fontsize'.
        fraction (tuple): Fraction of axis range to offset (fraction_x, fraction_y). Only used if offset_units is 'fraction'.
        ax (matplotlib.axes.Axes): The axes to annotate on. Needed for 'inches' or 'fraction' units.
    """

    # Get the default font size in points
    fontsize = plt.rcParams["font.size"] / 2 * 1.2

    # Default axis is required for 'inches' or 'fraction' units
    if ax is None and offset_units in ["inches", "fraction"]:
        raise ValueError(
            "Please provide an axes object when using 'inches' or 'fraction' units."
        )

    # Handle string-based offset directions
    if isinstance(offset, str):
        # Offset based on fontsize in the specified direction
        if offset_units == "fontsize":
            offset_x = (
                fontsize if offset == "right" else -fontsize if offset == "left" else 0
            )
            offset_y = (
                fontsize if offset == "up" else -fontsize if offset == "down" else 0
            )
        else:
            raise ValueError(
                "Please provide offset in 'fontsize' units or provide an explicit offset tuple."
            )

    elif isinstance(offset, tuple):
        # Custom offset in inches, points, or fraction of axis
        offset_x, offset_y = offset

        if offset_units == "inches":
            # Convert inches to display coordinates using the figure's DPI
            offset_x = ax.figure.dpi * offset_x
            offset_y = ax.figure.dpi * offset_y

        elif offset_units == "points":
            # Points are already the unit, so we don't modify the offset_x or offset_y
            pass

        elif offset_units == "fraction":
            # Fraction of axis, based on the axis data range
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()

            # Apply custom fraction offsets if provided
            offset_x = offset[0] * (x_range[1] - x_range[0])  # Fraction of x-axis range
            offset_y = offset[1] * (y_range[1] - y_range[0])  # Fraction of y-axis range

        else:
            raise ValueError(
                "Units must be 'fontsize', 'inches', 'points', or 'fraction'."
            )

    else:
        # No valid offset, set to zero
        offset_y = 0
        offset_x = 0

    # Calculate new object position based on offset
    new_x = xy[0] + offset_x
    new_y = xy[1] + offset_y

    return (new_x, new_y)
