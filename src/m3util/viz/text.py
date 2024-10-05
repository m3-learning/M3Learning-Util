import matplotlib.pyplot as plt
from matplotlib import (
    patheffects,
)
import numpy as np
import matplotlib
from m3util.util.kwargs import _filter_kwargs
from m3util.viz.positioning import obj_offset


def set_sci_notation_label(
    ax,
    axis="y",
    corner="bottom right",
    offset_points=None,
    scilimits=(0, 0),
    linewidth=None,
    stroke_color=None,
    write_to_axis=None,
):
    """
    Set scientific notation label on the specified axis of a plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to add the label to.
    axis (str, optional): The axis to apply the label to ('x' or 'y'). Defaults to 'y'.
    corner (str, optional): The corner of the plot to place the label. Defaults to 'bottom right'.
    offset_points (tuple, optional): Offset of the label in points. Defaults to None.
    scilimits (tuple, optional): The scientific notation limits. Defaults to (0, 0).
    linewidth (float, optional): The linewidth for the stroke effect. Defaults to None.
    stroke_color (str, optional): The stroke color for the text. Defaults to None.
    write_to_axis (matplotlib.axes.Axes, optional): The axis to write the label to. Defaults to None.

    Raises:
    ValueError: If an invalid corner position is provided.

    """
    # Allows for manually setting the axis to write to
    if write_to_axis is None:
        write_to_axis = ax

    # Set default offset points if not provided
    if offset_points is None:
        offset_points = (
            plt.rcParams["xtick.labelsize"] / 2,
            plt.rcParams["xtick.labelsize"] / 2,
        )

    # Apply scientific notation formatting to the specified axis
    ax.ticklabel_format(axis=axis, style="sci", scilimits=scilimits)

    # Hide the default offset text
    if axis == "x":
        ax.get_xaxis().get_offset_text().set_visible(False)
    else:
        ax.get_yaxis().get_offset_text().set_visible(False)

    # Get the ticks for the specified axis
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    ticks = ticks[ticks > 0]  # Filter out non-positive values to avoid log error
    if len(ticks) == 0:
        return  # No valid ticks to calculate the exponent, skip setting the label

    # Calculate the exponent for the scientific notation
    ax_max = max(ticks)
    exponent_axis = int(np.floor(np.log10(ax_max)))
    if exponent_axis == 0:
        return None  # No need to display exponent if it is 0
    exponent_text = r"$\times10^{%i}$" % exponent_axis

    # Define the corner positions
    corners = {
        "bottom left": (0, 0),
        "bottom right": (1, 0),
        "top left": (0, 1),
        "top right": (1, 1),
    }

    # Validate the corner position
    if corner not in corners:
        raise ValueError(
            "Invalid corner position. Choose from 'top left', 'top right', 'bottom left', 'bottom right'."
        )

    # Calculate the base position and offsets
    base_x, base_y = corners[corner]
    fig = ax.figure
    offset_x = offset_points[0] / fig.dpi / fig.get_size_inches()[0]
    offset_y = offset_points[1] / fig.dpi / fig.get_size_inches()[1]

    # Adjust offsets based on the corner position
    if "left" in corner:
        offset_x = offset_x
    else:
        offset_x = -offset_x
    if "bottom" in corner:
        offset_y = offset_y
    else:
        offset_y = -offset_y

    # Calculate the final text position
    text_x = base_x + offset_x
    text_y = base_y + offset_y

    # Configure path effects if linewidth and stroke color are provided
    path_effects_config = (
        [patheffects.withStroke(linewidth=linewidth, foreground=stroke_color)]
        if linewidth is not None and stroke_color is not None
        else None
    )

    # Add the scientific notation label to the plot
    write_to_axis.text(
        text_x,
        text_y,
        exponent_text,
        transform=ax.transAxes,
        ha="left" if "left" in corner else "right",
        va="bottom" if "bottom" in corner else "top",
        size=plt.rcParams["xtick.labelsize"] * 0.85,
        path_effects=path_effects_config,
        zorder=1000,
    )


def bring_text_to_front(fig, zorder=100):
    """
    Sets the zorder of all text objects in the Figure to the specified value.

    Parameters:
    fig (matplotlib.figure.Figure): The Figure object to modify.
    zorder (int, optional): The zorder value to set for text objects. Default is 100.
    """
    # Find all text objects in the figure
    text_objects = fig.findobj(match=matplotlib.text.Text)
    
    # Iterate over each text object and set its zorder
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
        "wb": dict(color="w", linewidth=0.75, foreground="k"),
        "b": dict(color="k", linewidth=0, foreground="k"), 
        "w": dict(color="w", linewidth=0, foreground="w"),
        "bw": dict(color="k", linewidth=0.75, foreground="w"),
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
            patheffects.withStroke(
                linewidth=formatting["linewidth"], foreground=formatting["foreground"]
            )
        ],
        color=formatting["color"],
        size=size,
        **kwargs,
    )

    text_.set_zorder(np.inf)
    
    return text_


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

    # Convert the desired text position in inches to a relative position (0 to 1)
    text_position_relative = (
        text_position_in_inches[0] / fig_size_inches[0],
        text_position_in_inches[1] / fig_size_inches[1],
    )

    # Add the text to the figure with the calculated relative position
    fig.text(text_position_relative[0], text_position_relative[1], text, **kwargs)



def line_annotation(ax, text, line_x, line_y, annotation_kwargs, zorder=100):
    """
    Annotate a line on a plot with text at its midpoint.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to add the annotation to.
    text (str): The text to annotate the line with.
    line_x (array-like): The x-coordinates of the line.
    line_y (array-like): The y-coordinates of the line.
    annotation_kwargs (dict): Additional keyword arguments for the annotation.
    zorder (int, optional): The zorder value for the annotation. Default is 100.
    """
    # Set text alignment
    ha = annotation_kwargs.get("ha", "center")
    va = annotation_kwargs.get("va", "center")

    # Calculate the midpoint of the line
    mid_x = np.mean(line_x)
    mid_y = np.mean(line_y)

    # Default offset for the text
    xytext = (0, 0)

    # Filter the annotation kwargs to remove any invalid arguments
    filtered_kwargs = _filter_kwargs(ax.annotate, annotation_kwargs)

    # Annotate the line at the midpoint with the specified text and properties
    ax.annotate(
        text,
        xy=(mid_x, mid_y),
        xycoords="data",
        xytext=obj_offset(xytext, **annotation_kwargs),
        textcoords="offset points",
        ha=ha,
        va=va,
        zorder=zorder,
        **filtered_kwargs,
    )
