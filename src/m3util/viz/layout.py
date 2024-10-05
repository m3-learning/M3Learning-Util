from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product
import numpy as np
from matplotlib import (
    path,
    patches,
    patheffects,
)
import PIL
import io
from m3util.viz.text import line_annotation

Path = path.Path
PathPatch = patches.PathPatch


def get_closest_point(x_data, y_data, value, axis="x"):
    """Get the closest point on a line plot to a provided x or y value.

    Args:
        x_data (array-like): Array of x data points.
        y_data (array-like): Array of y data points.
        value (float): The x or y value to find the closest point to.
        axis (str, optional): Specify which axis to use for finding the closest point.
            Must be 'x' or 'y'. Defaults to 'x'.

    Returns:
        tuple: (closest_x, closest_y) The closest point on the line plot.

    Raises:
        ValueError: If the axis is not 'x' or 'y', or if x_data and y_data have different lengths.
    """
    # Ensure x_data and y_data are NumPy arrays
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Check that x_data and y_data have the same length
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must have the same shape.")

    # Find the index of the closest point
    if axis == "x":
        idx = np.abs(x_data - value).argmin()
    elif axis == "y":
        idx = np.abs(y_data - value).argmin()
    else:
        raise ValueError("axis must be 'x' or 'y'")

    return x_data[idx], y_data[idx]


def plot_into_graph(axg, fig, colorbar_=True, clim=None, **kwargs):
    """Given an axes and figure, it will convert the figure to an image and plot it in

    Args:
        axg (matplotlib.axes.Axes): where you want to plot the figure
        fig (matplotlib.pyplot.figure()): figure you want to put into axes
    """
    img_buf = io.BytesIO()
    fig.savefig(img_buf, bbox_inches="tight", format="png")
    im = PIL.Image.open(img_buf)

    if clim is not None:
        ax_im = axg.imshow(im, clim=clim)
    else:
        ax_im = axg.imshow(im)

    if colorbar_:
        divider = make_axes_locatable(axg)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ax_im, cax=cax, **kwargs)

    img_buf.close()


def subfigures(
    nrows, ncols, size=(1.25, 1.25), gaps=(0.8, 0.33), figsize=None, **kwargs
):
    """
    Create subfigures with specified number of rows and columns.

    Parameters:
    nrows (int): Number of rows.
    ncols (int): Number of columns.
    size (tuple, optional): Size of each subfigures. Defaults to (1.25, 1.25).
    gaps (tuple, optional): Gaps between subfigures. Defaults to (.8, .33).
    figsize (tuple, optional): Size of the figure. Defaults to None.
    **kwargs: Additional keyword arguments.

    Returns:
    fig (Figure): The created figure.
    ax (list): List of axes objects.

    """
    if figsize is None:
        figsize = (size[0] * ncols + gaps[0] * ncols, size[1] * nrows + gaps[1] * nrows)

    # create a new figure with the specified size
    fig = plt.figure(figsize=figsize)

    ax = []

    for i, j in product(range(nrows), range(ncols)):
        rvalue = (nrows - 1) - j
        # calculate the position and size of each subfigure
        pos1 = [
            (size[0] * rvalue + gaps[0] * rvalue) / figsize[0],
            (size[1] * i + gaps[1] * i) / figsize[1],
            size[0] / figsize[0],
            size[1] / figsize[1],
        ]
        ax.append(fig.add_axes(pos1))

    ax.reverse()

    return fig, ax


def add_box(axs, pos, **kwargs):
    """
    Add a box to the axes.

    Parameters:
    axs (Axes): The axes to add the box to.
    pos (tuple): The position of the box in the form (xmin, ymin, xmax, ymax).
    **kwargs: Additional keyword arguments.

    """
    xmin, ymin, xmax, ymax = pos
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
    axs.add_patch(rect)


def inset_connector(fig, ax1, ax2, coord1=None, coord2=None, **kwargs):
    """
    Create a connection between two axes in a figure.

    Parameters:
    fig (Figure): The figure to add the connection to.
    ax1 (Axes): The first axes object.
    ax2 (Axes): The second axes object.
    coord1 (list, optional): The coordinates of the first connection point. Defaults to None.
    coord2 (list, optional): The coordinates of the second connection point. Defaults to None.
    **kwargs: Additional keyword arguments.

    """
    if coord1 is None:
        # Get the x and y limits of ax1
        coord1_xlim = ax1.get_xlim()
        coord1_ylim = ax1.get_ylim()

        # Calculate the coordinates of the first connection point
        coord1_l1 = (coord1_xlim[0], coord1_ylim[0])
        coord1_l2 = (coord1_xlim[0], coord1_ylim[1])
        coord1 = [coord1_l1, coord1_l2]

    if coord2 is None:
        # Get the x and y limits of ax2
        coord2_xlim = ax2.get_xlim()
        coord2_ylim = ax2.get_ylim()

        # Calculate the coordinates of the second connection point
        coord2_l1 = (coord2_xlim[0], coord2_ylim[0])
        coord2_l2 = (coord2_xlim[0], coord2_ylim[1])
        coord2 = [coord2_l1, coord2_l2]

    for p1, p2 in zip(coord1, coord2):
        # Create a connection between the two points
        con = ConnectionPatch(
            xyA=p1, xyB=p2, coordsA=ax1.transData, coordsB=ax2.transData, **kwargs
        )

        # Add the connection to the plot
        fig.add_artist(con)


def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Create a path patch and add it to the axes.

    Parameters:
    axes (Axes): The axes to add the path patch to.
    locations (tuple): The locations of the path in the form (x1, x2, y1, y2).
    facecolor (str): The face color of the path patch.
    edgecolor (str): The edge color of the path patch.
    linestyle (str): The line style of the path patch.
    lineweight (float): The line weight of the path patch.

    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    # Extract the vertices used to construct the path
    vertices = [
        (locations[0], locations[2]),
        (locations[1], locations[2]),
        (locations[1], locations[3]),
        (locations[0], locations[3]),
        (0, 0),
    ]
    vertices = np.array(vertices, float)
    # Make a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor, ls=linestyle, lw=lineweight
    )
    # Add the path to the axes
    axes.add_patch(pathpatch)


def layout_fig(graph, mod=None, figsize=None, layout="compressed", **kwargs):
    """
    Utility function that helps lay out many figures.

    Parameters:
    graph (int): Number of graphs.
    mod (int, optional): Value that assists in determining the number of rows and columns. Defaults to None.
    figsize (tuple, optional): Size of the figure. Defaults to None.
    layout (str, optional): Layout style of the subplots. Defaults to 'compressed'.
    **kwargs: Additional keyword arguments.

    Returns:
    tuple: Figure and axes.

    """
    # 10-5-2024 Stale code
    # # sets the kwarg values
    # for key, value in kwargs.items():
    #     exec(f"{key} = value")

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    if mod is None:
        # Select the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    if figsize is None:
        figsize = (3 * mod, 3 * (graph // mod + (graph % mod > 0)))

    # builds the figure based on the number of graphs and a selected number of columns
    fig, axes = plt.subplots(
        graph // mod + (graph % mod > 0), mod, figsize=figsize, layout=layout
    )

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return fig, axes[:graph]


def embedding_maps(data, image, colorbar_shown=True, c_lim=None, mod=None, title=None):
    """function that generates the embedding maps

    Args:
        data (array): embedding maps to plot
        image (array): raw image used for the sizing of the image
        colorbar_shown (bool, optional): selects if colorbars are shown. Defaults to True.
        c_lim (array, optional): sets the range for the color limits. Defaults to None.
        mod (int, optional): used to change the layout (rows and columns). Defaults to None.
        title (string, optional): Adds title to the image . Defaults to None.
    """
    fig, ax = layout_fig(data.shape[1], mod)

    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels("")
            ax.set_yticklabels("")

            # adds the colorbar
            if colorbar_shown is True:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="10%", pad=0.05)
                plt.colorbar(im, cax=cax, format="%.1e")

                # Sets the scales
                if c_lim is not None:
                    im.set_clim(c_lim)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16, y=1, horizontalalignment="center")

    try:
        fig.tight_layout()
    except RuntimeError:
        # Skip applying tight_layout if it causes conflicts
        pass


def imagemap(
    ax,
    data,
    colorbars=True,
    clim=None,
    divider_=True,
    cbar_number_format="%.1e",
    cmap_="viridis",
    **kwargs,
):
    """pretty way to plot image maps with standard formats

    Args:
        ax (ax): axes to write to
        data (array): data to write
        colorbars (bool, optional): selects if you want to show a colorbar. Defaults to True.
        clim (array, optional): manually sets the range of the colorbars. Defaults to None.
    """

    if data.ndim == 1:
        data = data.reshape(
            np.sqrt(data.shape[0]).astype(int), np.sqrt(data.shape[0]).astype(int)
        )

    cmap = plt.get_cmap(cmap_)

    if clim is None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data, vmin=clim[0], vmax=clim[1], clim=clim, cmap=cmap)

    ax.set_yticklabels("")
    ax.set_xticklabels("")
    ax.set_yticks([])
    ax.set_xticks([])

    if colorbars:
        if divider_:
            # adds the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            plt.colorbar(im, cax=cax, format=cbar_number_format)
        else:
            cb = plt.colorbar(im, fraction=0.046, pad=0.04, format=cbar_number_format)
            cb.ax.tick_params(labelsize=6, width=0.05)


def find_nearest(array, value, averaging_number):
    """computes the average of some n nearest neighbors

    Args:
        array (array): input array
        value (float): value to find closest to
        averaging_number (int): number of data points to use in averaging

    Returns:
        list: list of indexes of the nearest neighbors
    """
    idx = (np.abs(array - value)).argsort()[0:averaging_number]
    return idx


def combine_lines(*args):
    """
    Combine the lines and labels from multiple plots into a single legend.

    Args:
        *args: Variable number of arguments representing the plots.

    Returns:
        A tuple containing the combined lines and labels.

    Example:
        lines, labels = combine_lines(plot1, plot2, plot3)
    """

    lines = []
    labels = []

    for arg in args:
        # combine the two axes into a single legend
        line, label = arg.get_legend_handles_labels()
        lines += line
        labels += label

    return lines, labels


def scalebar(axes, image_size, scale_size, units="nm", loc="br"):
    """
    Adds a scalebar to figures.

    Parameters:
    axes (matplotlib.axes.Axes): The axes to add the scalebar to.
    image_size (int): The size of the image in nm.
    scale_size (str): The size of the scalebar in units of nm.
    units (str, optional): The units for the label. Defaults to "nm".
    loc (str, optional): The location of the label. Defaults to "br".
    """

    # Get the size of the image
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = (  # noqa: F841
        np.abs(np.int64(np.floor(x_lim[1] - x_lim[0]))),
        np.abs(np.int64(np.floor(y_lim[1] - y_lim[0]))),
    )
    # Compute the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1], np.int64(np.floor(image_size)))
    y_point = np.linspace(y_lim[0], y_lim[1], np.int64(np.floor(image_size)))

    # Set the location of the scalebar
    if loc == "br":
        x_start = x_point[np.int64(0.9 * image_size // 1)]
        x_end = x_point[np.int64((0.9 - fract) * image_size // 1)]
        y_start = y_point[np.int64(0.1 * image_size // 1)]
        y_end = y_point[np.int64((0.1 + 0.025) * image_size // 1)]
        y_label_height = y_point[np.int64((0.1 + 0.075) * image_size // 1)]
    elif loc == "tr":
        x_start = x_point[np.int64(0.9 * image_size // 1)]
        x_end = x_point[np.int64((0.9 - fract) * image_size // 1)]
        y_start = y_point[np.int64(0.9 * image_size // 1)]
        y_end = y_point[np.int64((0.9 - 0.025) * image_size // 1)]
        y_label_height = y_point[np.int64((0.9 - 0.075) * image_size // 1)]

    # Make the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], "w", "k", "-", 0.25)

    # Add the text label for the scalebar
    axes.text(
        (x_start + x_end) / 2,
        y_label_height,
        "{0} {1}".format(scale_size, units),
        size=6,
        weight="bold",
        ha="center",
        va="center",
        color="w",
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="k")],
    )


def Axis_Ratio(axes, ratio=1):
    """
    Set the aspect ratio of the axes to be proportional to the ratio of data ranges.

    Parameters:
    axes (matplotlib.axes.Axes): The axes object to set the aspect ratio for.
    ratio (float, optional): The desired aspect ratio. Defaults to 1.

    Returns:
    None
    """
    # Set aspect ratio to be proportional to the ratio of data ranges
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()

    xrange = xmax - xmin
    yrange = ymax - ymin

    axes.set_aspect(ratio * (xrange / yrange))


def get_axis_range(axs):
    """
    Return the minimum and maximum values of a Matplotlib axis.

    Parameters:
        axs (list): A list of Matplotlib axis objects.

    Returns:
        list: A list of the form [xmin, xmax, ymin, ymax], where xmin and xmax are the minimum and maximum values of the x axis, and ymin and ymax are the minimum and maximum values of the y axis.
    """

    def get_axis_range_(ax):
        """
        Return the minimum and maximum values of a Matplotlib axis.

        Parameters:
            ax (matplotlib.axis.Axis): The Matplotlib axis object to get the range of.

        Returns:
            tuple: A tuple of the form (xmin, xmax, ymin, ymax), where xmin and xmax are the minimum and maximum values of the x axis, and ymin and ymax are the minimum and maximum values of the y axis.
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        return xmin, xmax, ymin, ymax

    xmin = None
    xmax = None
    ymin = None
    ymax = None

    for ax in axs:
        ax_xmin, ax_xmax, ax_ymin, ax_ymax = get_axis_range_(ax)
        try:
            xmin = np.min(xmin, ax_xmin)
            xmax = np.max(xmax, ax_xmax)
            ymin = np.min(ymin, ax_ymin)
            ymax = np.max(ymax, ax_ymax)
        except:  # noqa: E722
            xmin = ax_xmin
            xmax = ax_xmax
            ymin = ax_ymin
            ymax = ax_ymax

    return [xmin, xmax, ymin, ymax]


def set_axis(axs, range):
    """
    Set the x and y axis limits for each axis in the given list.

    Parameters:
    axs (list): A list of matplotlib axes objects.
    range (list): A list containing the x and y axis limits in the format [xmin, xmax, ymin, ymax].

    Returns:
    None
    """
    for ax in axs:
        ax.set_xlim(range[0], range[1])
        ax.set_ylim(range[2], range[3])


def add_scalebar(ax, scalebar_):
    """Adds a scalebar to the figure

    Args:
        ax (axes): axes to add the scalebar to
        scalebar_ (dict): dictionary containing the scalebar information
    """

    if scalebar_ is not None:
        scalebar(
            ax, scalebar_["width"], scalebar_["scale length"], units=scalebar_["units"]
        )


def get_axis_pos_inches(fig, ax):
    """gets the position of the axis in inches

    Args:
        fig (matplotlib.Figure): figure where the plot is located
        ax (matplotlib.axes): axes on the plot

    Returns:
        array: the position of the center bottom of the axis in inches
    """

    # Get the bounding box of the axis in normalized coordinates (relative to the figure)
    axis_bbox = ax.get_position()

    # Calculate the center bottom point of the axis in normalized coordinates
    center_bottom_x = axis_bbox.x0 + axis_bbox.width / 2
    center_bottom_y = axis_bbox.y0

    # Convert the center bottom point from normalized coordinates to display units
    center_bottom_display = fig.transFigure.transform(
        (center_bottom_x, center_bottom_y)
    )

    return center_bottom_display / fig.dpi


def layout_subfigures_inches(size, subfigures_dict, margin_pts=20):
    """
    Creates a matplotlib figure with subfigures arranged based on positions in inches,
    and manually adds margins (in points) to accommodate axis labels and titles.
    Allows skipping margins for specific subfigures using a 'skip_margin' flag,
    which defaults to False (i.e., margins are applied unless specified).

    Parameters:
    - size: tuple of (width, height) for the overall figure size in inches.
    - subfigures_dict: dictionary where keys are subfigure names and values are dictionaries
                       containing 'position' as a tuple (x, y, width, height) in inches,
                       'plot_func' as the function to create the specific subfigure, and
                       'skip_margin' (optional) as a boolean to indicate whether to skip margins.
    - margin_pts: margin in points (72 points = 1 inch) for labels, tick marks, and titles.

    Returns:
    - fig: the matplotlib figure object.
    - axes_dict: a dictionary where keys are the subfigure names and values are the corresponding axes.
    """
    # Convert points to inches (72 points = 1 inch)
    margin_inch = margin_pts / 72.0

    # Create the main figure with the specified size
    fig = plt.figure(figsize=size)
    axes_dict = {}

    for name, subfig_data in subfigures_dict.items():
        position = subfig_data["position"]  # (x, y, width, height) in inches
        skip_margin = subfig_data.get("skip_margin", False)  # Defaults to False
        right = subfig_data.get("right", False)  # Defaults to False

        if right == True:
            multiple = 2
        else:
            multiple = 1

        # If skip_margin is False, apply the margin
        if not skip_margin:
            left = (position[0] + margin_inch) / size[0]
            bottom = (position[1] + margin_inch) / size[1]
            width = (position[2] - multiple * margin_inch) / size[0]
            height = (position[3] - multiple * margin_inch) / size[1]
        else:
            # No margin adjustments
            left = position[0] / size[0]
            bottom = position[1] / size[1]
            width = position[2] / size[0]
            height = position[3] / size[1]

        # Add an axes to the figure at the specified location
        ax = fig.add_axes([left, bottom, width, height])

        # Store the axes in the dictionary with the corresponding name
        axes_dict[name] = ax

    return fig, axes_dict


class FigDimConverter:
    """class to convert between relative and inches dimensions of a figure"""

    def __init__(self, figsize):
        """initializes the class

        Args:
            figsize (tuple): figure size in inches
        """

        self.fig_width = figsize[0]
        self.fig_height = figsize[1]

    def to_inches(self, x):
        """Converts position from relative to inches

        Args:
            x (tuple): position in relative coordinates (left, bottom, width, height)

        Returns:
            tuple: position in inches (left, bottom, width, height)
        """

        return (
            x[0] * self.fig_width,
            x[1] * self.fig_height,
            x[2] * self.fig_width,
            x[3] * self.fig_height,
        )

    def to_relative(self, x):
        """Converts position from inches to relative

        Args:
            x (tuple): position in inches (left, bottom, width, height)

        Returns:
            tuple: position in relative coordinates (left, bottom, width, height)
        """

        return (
            x[0] / self.fig_width,
            x[1] / self.fig_height,
            x[2] / self.fig_width,
            x[3] / self.fig_height,
        )


def get_zorders(fig):
    """
    Retrieves the z-order of all objects in a given Matplotlib figure.

    Parameters:
    - fig: Matplotlib Figure object

    Returns:
    - List of tuples containing (object description, zorder)
    """
    zorder_list = []

    # Iterate over all axes in the figure
    for ax in fig.get_axes():
        # Check items in axes (lines, text, etc.)
        for item in ax.get_children():
            desc = str(item)  # Get a string description of the item
            try:
                # Append description and zorder to the list
                zorder_list.append((desc, item.get_zorder()))
            except AttributeError:
                # Not all elements have a zorder attribute
                continue

        # Check zorder for major and minor ticks
        for axis in [ax.xaxis, ax.yaxis]:
            for which in ["major", "minor"]:
                ticks = axis.get_ticklabels(which=which)
                for tick in ticks:
                    zorder_list.append(
                        (f"Tick Label ({tick.get_text()})", tick.get_zorder())
                    )

    return zorder_list


def draw_line_with_text(
    ax,
    x_data,
    y_data,
    value,
    axis="x",
    span="full",
    text="",
    zorder=2,
    line_kwargs={},
    annotation_kwargs={},
):
    """
    Draw a horizontal or vertical line on a plot, either spanning the axis or between the closest two data points,
    with optional text offset by a fixed number of points perpendicular to the line.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        x_data (array-like): The x data points of the plot.
        y_data (array-like): The y data points of the plot.
        value (float): The x or y value at which to draw the line.
        axis (str, optional): Specifies whether to draw a vertical ('x') or horizontal ('y') line (default is 'x').
        span (str, optional): Specifies whether the line spans the full axis ('full') or between the closest two data points ('data').
        text (str, optional): Text to place near the line.
        zorder (int or float, optional): The z-order of the line and text (default is 2).
        **kwargs: Additional keyword arguments passed to the line drawing function (e.g., color, linewidth).

    Raises:
        ValueError: If invalid values are provided for axis or span.
    """
    # Ensure x_data and y_data are NumPy arrays
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Validate axis parameter
    if axis not in ["x", "y"]:
        raise ValueError("axis must be 'x' or 'y'")

    # Validate span parameter
    if span not in ["full", "axis", "data"]:
        raise ValueError("span must be 'full' or 'data'")

    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Initialize line start and end points
    if span == "full":
        if axis == "x":
            # Vertical line spanning full y-axis
            line_x = [value, value]
            line_y = [y_min, y_max]
        else:
            # Horizontal line spanning full x-axis
            line_x = [x_min, x_max]
            line_y = [value, value]
    elif span == "axis":
        connect_to = line_kwargs.get("connect_to", "left")
        line_x, line_y = span_to_axis(ax, value, x_data, y_data, connect_to=connect_to)
    else:
        # Span between closest data points
        if axis == "x":
            # Vertical line between two y-values at closest x-values to 'value'
            idx_below = np.where(x_data <= value)[0]
            idx_above = np.where(x_data >= value)[0]

            if idx_below.size == 0 or idx_above.size == 0:
                raise ValueError("Value is outside the range of x_data.")

            idx1 = idx_below[-1]
            idx2 = idx_above[0]

            y1 = y_data[idx1]
            y2 = y_data[idx2]

            line_x = [value, value]
            line_y = [y1, y2]
        else:
            # Horizontal line between two x-values at closest y-values to 'value'
            idx_below = np.where(y_data <= value)[0]
            idx_above = np.where(y_data >= value)[0]

            if idx_below.size == 0 or idx_above.size == 0:
                raise ValueError("Value is outside the range of y_data.")

            idx1 = idx_below[-1]
            idx2 = idx_above[0]

            x1 = x_data[idx1]
            x2 = x_data[idx2]

            line_x = [x1, x2]
            line_y = [value, value]

    # Set zorder in line properties
    line_kwargs["zorder"] = zorder

    # Draw the line
    ax.plot(line_x, line_y, **line_kwargs)

    # Add text if provided
    if text:
        line_annotation(ax, text, line_x, line_y, annotation_kwargs, zorder=zorder)

def span_to_axis(ax, value, x_data, y_data, connect_to="left"):
    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Span between closest data points
    if connect_to == "left" or connect_to == "right":
        x_, y_ = get_closest_point(x_data, y_data, value, axis="x")

        # Determine which axis to connect to
        if connect_to == "left":
            line_x = [x_, x_min]  # Connect to left y-axis
            line_y = [y_, y_]
        elif connect_to == "right":
            line_x = [x_, x_max]  # Connect to right y-axis
            line_y = [y_, y_]
        else:
            raise ValueError("Invalid connect_to value. Choose 'left', 'right'")
    elif connect_to == "bottom" or connect_to == "top":
        x_, y_ = get_closest_point(x_data, y_data, value, axis="y")

        if connect_to == "bottom":
            line_x = [x_, x_]
            line_y = [y_, y_min]  # Connect to bottom x-axis
        elif connect_to == "top":
            line_x = [x_, x_]
            line_y = [y_, y_max]  # Connect to top x-axis
        else:
            raise ValueError("Invalid connect_to value. Choose 'bottom', 'top'")
    else:
        raise ValueError(
            "Invalid connect_to value. Choose 'left', 'right', 'bottom', 'top'"
        )
    return line_x, line_y


# Mock line_annotation since it's used internally
def mock_line_annotation(ax, text, line_x, line_y, annotation_kwargs, zorder=2):
    # A simple mock to bypass actual text annotation
    pass