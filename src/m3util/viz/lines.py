
def draw_lines(ax, x_values, y_values, style=None):
    """
    Draws lines on a matplotlib axis based on x and y values.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw the lines.
        x_values (list or array): A list or array of x-values for the line.
        y_values (list or array): A list or array of y-values for the line.
        style (dict, optional): A dictionary specifying line style properties.
                                Keys can include 'color', 'linewidth', 'linestyle', etc.

    Returns:
        matplotlib.lines.Line2D: The line artist object.
    """
    # Default style if none is provided
    if style is None:
        style = {}

    # Draw the line with the provided style
    line = ax.plot(x_values, y_values, **style)

    return line
