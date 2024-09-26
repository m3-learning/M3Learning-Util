import matplotlib.pyplot as plt
from matplotlib import (
    path,
    patches,
    patheffects,
)
import numpy as np

def set_sci_notation_label(
    ax, axis="y", corner="bottom right", offset_points=None, scilimits=(0, 0), linewidth=None, stroke_color=None
):
    """
    Formats the specified axis to use scientific notation and places the exponent
    label at a specified position relative to a corner of the axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object to modify.
        axis (str): Axis to format ('x' or 'y'). Defaults to 'y'.
        corner (str): Corner of the axis to position the exponent label.
                      Options are 'top left', 'top right', 'bottom left', 'bottom right'.
                      Defaults to 'bottom right'.
        offset_points (tuple): Offset in points (x_offset, y_offset) from the specified corner.
                               Defaults to (5, 5).
        scilimits (tuple): The range of exponents where scientific notation is used.
        linewidth (int): The width of the stroke around the text. Defaults to None.
        stroke_color (str): The color of the stroke around the text. Defaults to None.
    """

    if offset_points is None:
        # Default offset in points
        offset_points = (
            plt.rcParams["xtick.labelsize"] / 2,
            plt.rcParams["xtick.labelsize"] / 2,
        )

    # Apply scientific notation formatting to the specified axis
    ax.ticklabel_format(axis=axis, style="sci", scilimits=scilimits)

    # Hide the default offset text (the default scientific notation label)
    if axis == "x":
        ax.get_xaxis().get_offset_text().set_visible(False)
    else:
        ax.get_yaxis().get_offset_text().set_visible(False)

    # Get the maximum value from the ticks to determine the exponent
    if axis == "x":
        ticks = ax.get_xticks()
    else:
        ticks = ax.get_yticks()

    # Avoid log of zero by filtering out non-positive ticks
    ticks = ticks[ticks != 0]
    if len(ticks) == 0:
        exponent_axis = 0
    else:
        ax_max = max(abs(ticks))
        exponent_axis = int(np.floor(np.log10(ax_max)))

    # Create the exponent label using LaTeX formatting
    exponent_text = r"$\times10^{%i}$" % (exponent_axis)

    # Define corner positions in axis fraction coordinates
    corners = {
        "bottom left": (0, 0),
        "bottom right": (1, 0),
        "top left": (0, 1),
        "top right": (1, 1),
    }
    if corner not in corners:
        raise ValueError(
            "Invalid corner position. Choose from 'top left', 'top right', 'bottom left', 'bottom right'."
        )

    # Get the base position for the text placement
    base_x, base_y = corners[corner]

    # Convert offset from points to axis fraction using the inverse transformation
    # We need to account for the figure's DPI and size
    fig = ax.figure
    offset_x = offset_points[0] / fig.dpi / fig.get_size_inches()[0]
    offset_y = offset_points[1] / fig.dpi / fig.get_size_inches()[1]

    # Adjust the position based on the corner
    if "left" in corner:
        offset_x = offset_x
    else:
        offset_x = -offset_x
    if "bottom" in corner:
        offset_y = offset_y
    else:
        offset_y = -offset_y

    # Final position after applying the offset
    text_x = base_x + offset_x
    text_y = base_y + offset_y
    
    if linewidth is not None and stroke_color is not None:
        pass
    elif stroke_color is not None:
        linewidth = .5
    else:
        linewidth = None
        
    
    path_effects = (
    [patheffects.withStroke(linewidth=linewidth, foreground=stroke_color)]
        )   

    # Use ax.text() instead of annotate to directly place the text
    ax.text(
        text_x,
        text_y,
        exponent_text,
        transform=ax.transAxes,
        ha="left" if "left" in corner else "right",
        va="bottom" if "bottom" in corner else "top",
        size=plt.rcParams["xtick.labelsize"]*.6,
        path_effects=path_effects,
        zorder=1000,
    )


def bring_text_to_front(fig, zorder=100):
    """
    Sets the zorder of all text objects in the Figure to the specified value.

    Args:
        fig (matplotlib.figure.Figure): The Figure object to modify.
        zorder (int, optional): The zorder value to set for text objects. Default is 5.
    """
    # Find all text objects in the figure
    text_objects = fig.findobj(match=matplotlib.text.Text)
    for text in text_objects:
        text.set_zorder(zorder)


def labelfigs(
    axes,
    number=None,
    style="wb",
    loc="tl",
    string_add="",
    size=8,
    text_pos="center",
    inset_fraction=(0.15, 0.15),
    **kwargs,
):
    """
    Add labels to figures.

    Parameters:
    axes (Axes): The axes to add the labels to.
    number (int, optional): The number to be added as a label. Defaults to None.
    style (str, optional): The style of the label. Defaults to "wb".
    loc (str, optional): The location of the label. Defaults to "tl".
    string_add (str, optional): Additional string to be added to the label. Defaults to "".
    size (int, optional): The font size of the label. Defaults to 8.
    text_pos (str, optional): The position of the label text. Defaults to "center".
    inset_fraction (tuple, optional): The fraction of the axes to inset the label. Defaults to (0.15, 0.15).
    **kwargs: Additional keyword arguments.

    Returns:
    Text: The created text object.

    Raises:
    ValueError: If an invalid position is provided.

    """

    # initializes an empty string
    text = ""

    # Sets up various color options
    formatting_key = {
        "wb": dict(color="w", linewidth=0.75),
        "b": dict(color="k", linewidth=0),
        "w": dict(color="w", linewidth=0),
    }

    # Stores the selected option
    formatting = formatting_key[style]
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]

    if loc == "tl":
        x, y = xlim[0] + x_inset, ylim[1] - y_inset
    elif loc == "tr":
        x, y = xlim[1] - x_inset, ylim[1] - y_inset
    elif loc == "bl":
        x, y = xlim[0] + x_inset, ylim[0] + y_inset
    elif loc == "br":
        x, y = xlim[1] - x_inset, ylim[0] + y_inset
    elif loc == "ct":
        x, y = (xlim[0] + xlim[1]) / 2, ylim[1] - y_inset
    elif loc == "cb":
        x, y = (xlim[0] + xlim[1]) / 2, ylim[0] + y_inset
    else:
        raise ValueError(
            "Invalid position. Choose from 'tl', 'tr', 'bl', 'br', 'ct', or 'cb'."
        )

    text += string_add

    if number is not None:
        text += number_to_letters(number)

    text_ = axes.text(
        x,
        y,
        text,
        va=text_pos,
        ha="center",
        path_effects=[
            patheffects.withStroke(linewidth=formatting["linewidth"], foreground="k")
        ],
        color=formatting["color"],
        size=size,
        **kwargs,
    )

    text_.set_zorder(np.inf)


def number_to_letters(num):
    """
    Convert a number to a string representation using letters.

    Parameters:
    num (int): The number to convert.

    Returns:
    str: The string representation of the number.

    """
    letters = ""
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1  # decrease num by 1 because we have processed the current digit
    return letters


def add_text_to_figure(fig, text, text_position_in_inches, **kwargs):
    """
    Add text to a figure at a specified position.

    Parameters:
    fig (Figure): The figure to add the text to.
    text (str): The text to be added.
    text_position_in_inches (tuple): The position of the text in inches.
    **kwargs: Additional keyword arguments.

    """
    # Get the figure size in inches and dpi
    fig_size_inches = fig.get_size_inches()
    fig_dpi = fig.get_dpi()

    # Convert the desired text position in inches to a relative position (0 to 1)
    text_position_relative = (
        text_position_in_inches[0] / fig_size_inches[0],
        text_position_in_inches[1] / fig_size_inches[1],
    )

    # Add the text to the figure with the calculated relative position
    fig.text(text_position_relative[0], text_position_relative[1], text, **kwargs)
