import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from m3util.viz.layout import get_closest_point
from m3util.viz.positioning import obj_offset
from m3util.viz.lines import draw_lines
import matplotlib.patheffects as path_effects


def draw_ellipse_with_arrow(
    ax,
    x_data,
    y_data,
    value,
    width,
    height,
    axis="x",
    line_direction="horizontal",
    arrow_position="top",
    arrow_length_frac=0.3,
    color="blue",
    linewidth=2,
    arrow_props=None,
    ellipse_props=None,
    arrow_direction="positive",
):
    """
    Draws an ellipse at a specified location on a line plot and adds an arrow originating 
    from the ellipse.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis where the ellipse and arrow are to be drawn.
        x_data (array-like): X-axis data points of the line plot.
        y_data (array-like): Y-axis data points of the line plot.
        value (float): The x or y value where the ellipse should be placed, depending on the `axis`.
        width (float): The width of the ellipse, specified as a fraction of the x-axis range.
        height (float): The height of the ellipse, specified as a fraction of the y-axis range.
        axis (str, optional): The axis ('x' or 'y') used to find the closest point for placing the ellipse.
                              Default is 'x'.
        line_direction (str, optional): Defines the orientation of the line ('horizontal' or 'vertical') to 
                                        which the ellipse and arrow are aligned. Default is 'horizontal'.
        arrow_position (str, optional): The position of the arrow relative to the ellipse ('top' or 'bottom').
                                        Default is 'top'.
        arrow_length_frac (float, optional): The length of the arrow as a fraction of the axis range. 
                                             Default is 0.3.
        color (str, optional): The color of both the ellipse and the arrow. Default is 'blue'.
        linewidth (float, optional): The line width of the ellipse outline. Default is 2.
        arrow_props (dict, optional): Additional properties to customize the arrow's appearance, passed as 
                                      a dictionary. Default is None.
        ellipse_props (dict, optional): Additional properties to customize the ellipse's appearance, passed 
                                        as a dictionary. Default is None.
        arrow_direction (str, optional): The direction of the arrow ('positive' or 'negative'). Determines 
                                         the direction of arrow relative to the axis. Default is 'positive'.

    Raises:
        ValueError: Raised if invalid values are provided for `axis`, `line_direction`, or `arrow_position`.

    Example:
        draw_ellipse_with_arrow(ax, x_data, y_data, value=5.0, width=0.1, height=0.05, axis='x')
    """
    # Ensure x_data and y_data are converted to NumPy arrays
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Check if x_data and y_data have the same length
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must have the same shape.")

    # Find the closest point on the line (either along x-axis or y-axis)
    ellipse_center = get_closest_point(x_data, y_data, value, axis=axis)

    # Get the axis limits to properly scale the ellipse dimensions and arrow length
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Compute the arrow length based on the direction of the line
    if line_direction == "horizontal":
        arrow_length = arrow_length_frac * (x_max - x_min)
        if arrow_direction == "negative":
            arrow_length = -arrow_length
    elif line_direction == "vertical":
        arrow_length = arrow_length_frac * (y_max - y_min)
        if arrow_direction == "negative":
            arrow_length = -arrow_length
    else:
        raise ValueError("line_direction must be 'horizontal' or 'vertical'.")

    # Scale the width and height of the ellipse based on the axis limits
    width_scaled = width * (x_max - x_min)
    height_scaled = height * (y_max - y_min)

    # Define default ellipse properties and update with user-provided values if applicable
    default_ellipse_props = {"edgecolor": color, "facecolor": "none", "lw": linewidth}
    if ellipse_props:
        default_ellipse_props.update(ellipse_props)

    # Draw the ellipse at the computed location
    ellipse = Ellipse(
        xy=ellipse_center,
        width=width_scaled,
        height=height_scaled,
        **default_ellipse_props,
    )
    ax.add_patch(ellipse)

    # Determine the arrow start and end points based on line direction and arrow position
    if line_direction == "horizontal":
        if arrow_position == "top":
            start_point = (ellipse_center[0], ellipse_center[1] + height_scaled / 2)
            end_point = (
                ellipse_center[0] + arrow_length,
                ellipse_center[1] + height_scaled / 2,
            )
        elif arrow_position == "bottom":
            start_point = (ellipse_center[0], ellipse_center[1] - height_scaled / 2)
            end_point = (
                ellipse_center[0] + arrow_length,
                ellipse_center[1] - height_scaled / 2,
            )
        else:
            raise ValueError("arrow_position must be 'top' or 'bottom'.")
    elif line_direction == "vertical":
        if arrow_position == "top":
            start_point = (ellipse_center[0] + width_scaled / 2, ellipse_center[1])
            end_point = (
                ellipse_center[0] + width_scaled / 2,
                ellipse_center[1] + arrow_length,
            )
        elif arrow_position == "bottom":
            start_point = (ellipse_center[0] - width_scaled / 2, ellipse_center[1])
            end_point = (
                ellipse_center[0] - width_scaled / 2,
                ellipse_center[1] + arrow_length,
            )
        else:
            raise ValueError("arrow_position must be 'top' or 'bottom'.")
    else:
        raise ValueError("line_direction must be 'horizontal' or 'vertical'.")

    # Define default arrow properties and update with user-provided values if applicable
    default_arrow_props = {
        "facecolor": color,
        "width": 2,
        "headwidth": 10,
        "headlength": 10,
        "linewidth": 0,
    }
    if arrow_props:
        default_arrow_props.update(arrow_props)

    # Draw the arrow on the plot using the defined start and end points
    ax.annotate("", xy=end_point, xytext=start_point, arrowprops=default_arrow_props)


def place_text_in_inches(
    fig,
    text,
    x_inch,
    y_inch,
    angle,
    stroke_width=None,
    stroke_color="black",
    **textprops,
):
    """
    Places a text element on a matplotlib figure at a specific position given in inches, 
    with options for rotating the text and adding a stroke (outline) for enhanced visibility.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure on which to place the text.
        text (str): The string to display as text.
        x_inch (float): The x-coordinate in inches, relative to the left of the figure.
        y_inch (float): The y-coordinate in inches, relative to the bottom of the figure.
        angle (float): The angle to rotate the text, in degrees.
        stroke_width (int, optional): The width of the text outline (stroke) in points. Default is None, meaning no stroke.
        stroke_color (str, optional): The color of the text outline (stroke). Default is 'black'.
        **textprops: Additional keyword arguments for customizing the text properties 
                    (e.g., fontsize, color, fontweight, etc.).

    Returns:
        matplotlib.text.Text: The text artist object added to the figure.

    Notes:
        - The position is specified in inches, and the function converts it to pixel-based display coordinates.
        - Stroke (outline) is achieved using `matplotlib.patheffects` to enhance text visibility.
        - The figure is redrawn after adding the text to ensure the update appears immediately.

    Example:
        place_text_in_inches(fig, "Sample Text", 2, 3, angle=45, fontsize=12, color="red")
    """
    # Convert from inches to display coordinates (pixel units) using the figure's DPI scaling transformation
    display_coords = fig.dpi_scale_trans.transform((x_inch, y_inch))

    # Add the text at the computed display coordinates
    text_artist = plt.text(
        display_coords[0],  # X-coordinate in pixel coordinates
        display_coords[1],  # Y-coordinate in pixel coordinates
        text,               # Text string to be displayed
        horizontalalignment="center",  # Center text horizontally
        verticalalignment="center",    # Center text vertically
        transform=None,     # Coordinates are already in display space, no additional transform needed
        rotation=angle,     # Rotate the text to the specified angle
        **textprops,        # Pass additional text properties such as fontsize, color, etc.
    )

    # If stroke (outline) is specified, apply the stroke effect using PathEffects
    if stroke_width is not None:
        text_artist.set_path_effects(
            [
                path_effects.Stroke(linewidth=stroke_width, foreground=stroke_color),  # Stroke with given width and color
                path_effects.Normal(),  # Draw the text over the stroke to maintain readability
            ]
        )

    # Redraw the figure to ensure the new text is displayed immediately
    fig.canvas.draw()

    return text_artist


def place_text_points(
    fig,
    text,
    x,
    y,
    angle,
    ax=None,
    stroke_width=None,
    stroke_color="black",
    **textprops,
):
    """
    Places a text element on a specified position in axis coordinates (or figure coordinates if no axis is provided) 
    with options for rotating the text and adding a stroke (outline) for enhanced visibility.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure where the text will be placed.
        text (str): The text string to be displayed.
        x (float): The x-coordinate for placing the text in axis coordinates (or figure coordinates if `ax` is None).
        y (float): The y-coordinate for placing the text in axis coordinates (or figure coordinates if `ax` is None).
        angle (float): The angle to rotate the text, in degrees.
        ax (matplotlib.axes.Axes, optional): The axes on which to place the text. If None, the text is placed directly on the figure.
        stroke_width (int, optional): The width of the stroke (outline) around the text, in points. Default is None (no stroke).
        stroke_color (str, optional): The color of the stroke (outline). Default is 'black'.
        **textprops: Additional keyword arguments to customize text properties (e.g., `fontsize`, `color`, `fontweight`, etc.).

    Returns:
        matplotlib.text.Text: The `Text` artist object added to the figure or axis.

    Notes:
        - If no axis (`ax`) is provided, the text is placed on the figure, and the coordinates are relative to the figure.
        - Stroke (outline) can be applied to the text using `path_effects` for improved visibility, particularly over complex backgrounds.
        - The figure is redrawn immediately after placing the text to ensure the new text is displayed.

    Example:
        place_text_points(fig, "Label", x=0.5, y=0.5, angle=45, ax=ax, fontsize=12, color="red", stroke_width=1.5)
    """

    # Ensure the axis is provided; otherwise, the figure will be used
    if ax is None:
        ax = fig.gca()  # Get current axis if not provided

    # Create the text artist on the specified axis or figure
    text_artist = ax.text(
        x,  # x-coordinate in axis coordinates
        y,  # y-coordinate in axis coordinates
        text,  # Text string to display
        horizontalalignment="center",  # Center the text horizontally
        verticalalignment="center",    # Center the text vertically
        rotation=angle,  # Apply the specified rotation angle to the text
        **textprops,  # Pass additional text properties such as fontsize, color, etc.
    )

    # Apply a stroke (outline) around the text if stroke_width is specified
    if stroke_width is not None:
        text_artist.set_path_effects(
            [
                path_effects.Stroke(linewidth=stroke_width, foreground=stroke_color),  # Set stroke width and color
                path_effects.Normal(),  # Draw the text over the stroke to maintain readability
            ]
        )

    # Redraw the figure to ensure the text and any applied effects are rendered
    fig.canvas.draw()

    return text_artist


# def place_text(fig, text, x, y, angle, ax=None, **textprops):

#     if ax is None:
#         # Convert from inches to display coordinates (pixels) using the figure's dpi scale transform
#         display_coords = fig.dpi_scale_trans.transform((x, y))
#         x_, y_ = display_coords[0], display_coords[1]
#     else:
#         x_, y_ = x, y

#     text_artist = ax.text(
#         x_,  # x-coordinate in axis coordinates
#         y_,  # y-coordinate in axis coordinates
#         text,  # Text string to display
#         horizontalalignment="center",  # Horizontal alignment of the text
#         verticalalignment="center",  # Vertical alignment of the text
#         rotation=angle,  # Rotation angle of the text
#         **textprops,  # Additional text properties
#     )

#     # Trigger the figure redraw to update the display with the new text
#     fig.canvas.draw()

#     return text_artist


def shift_object_in_points(ax, position_axis, direction_vector, n_points):
    """
    Shifts a position by a specified number of points along a given vector direction, returning the new position in axis coordinates.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes on which to perform the shift.
        position_axis (tuple of float): The starting position in axis coordinates as (x, y).
        direction_vector (tuple of float): The direction vector for the shift as (dx, dy).
        n_points (float): The number of points to shift along the direction vector.

    Returns:
        tuple of float: The new position in axis coordinates as (x, y).
    """
    # Convert the starting position from axis coordinates to display (points) coordinates
    position_display = ax.transData.transform(position_axis)

    # Normalize the direction vector to get the unit direction
    direction_vector = np.array(direction_vector)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the shift vector in points along the specified direction
    shift_vector_pts = direction_vector * n_points

    # Apply the shift to the original position in display coordinates
    new_position_display = position_display + shift_vector_pts

    # Convert the new display coordinates back to axis (data) coordinates
    new_position_axis = ax.transData.inverted().transform(new_position_display)

    return tuple(new_position_axis)


def shift_object_in_inches(fig, position_inch, direction_vector, n_points):
    """
    Shifts a position by a specified number of points along a given vector direction, returning the new position in inches.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure, used to get the DPI for point-to-inch conversion.
        position_inch (tuple of float): The starting position in inches as (x, y).
        direction_vector (tuple of float): The direction vector for the shift as (dx, dy).
        n_points (float): The number of points to shift along the direction vector.

    Returns:
        tuple of float: The new position in inches as (x, y).
    """
    # Normalize the direction vector to get the unit direction
    direction_vector = np.array(direction_vector)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Get the figure's DPI to convert points to inches
    dpi = fig.dpi
    points_per_inch = 72  # 1 inch = 72 points

    # Convert the shift in points to inches
    shift_inch = n_points / points_per_inch

    # Calculate the shift vector in inches along the specified direction
    shift_vector_inch = direction_vector * shift_inch

    # Apply the shift to the original position (which is already in inches)
    new_position_inch = np.array(position_inch) + shift_vector_inch

    return tuple(new_position_inch)


def get_perpendicular_vector(point1, point2, clockwise=False):
    """
    Computes the perpendicular vector to the vector defined by two points.

    Args:
        point1 (tuple of float): The first endpoint as (x1, y1).
        point2 (tuple of float): The second endpoint as (x2, y2).
        clockwise (bool, optional): If True, returns the clockwise perpendicular vector.
            Otherwise, returns the counterclockwise perpendicular vector. Defaults to False.

    Returns:
        tuple of float: The perpendicular vector as (dx, dy).
    """
    # Calculate the direction vector from point1 to point2
    x1, y1 = point1
    x2, y2 = point2
    direction_vector = np.array([x2 - x1, y2 - y1])

    # Compute the perpendicular vector
    if clockwise:
        # Rotate the direction vector by -90 degrees (clockwise)
        perpendicular_vector = np.array(
            [direction_vector[1], -direction_vector[0]]
        )  # Clockwise rotation
    else:
        # Rotate the direction vector by +90 degrees (counterclockwise)
        perpendicular_vector = np.array(
            [-direction_vector[1], direction_vector[0]]
        )  # Counterclockwise rotation

    return tuple(perpendicular_vector)


class DrawArrow:
    def __init__(
        self,
        fig,
        start_pos,
        end_pos,
        text=None,
        ax=None,
        text_position="center",
        text_alignment="center",
        vertical_text_displacement=None,
        units="inches",
        scale="figure fraction",
        arrowprops=dict(arrowstyle="->"),
        textprops=None,
        halo={},
    ):
        self._ax = ax

        if ax is None:
            self.ax = plt
        else:
            self.ax = ax

        # Initialize object properties
        self.fig = fig
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.text = text
        self.text_position = text_position
        self.text_alignment = text_alignment
        self.vertical_text_displacement = vertical_text_displacement
        self.units = units
        self.scale = scale
        self.arrowprops = arrowprops
        self.textprops = textprops
        self.halo = halo
        self.set_vertical_text_displacement()

    def set_vertical_text_displacement(self):
        """
        Sets the vertical displacement for the text based on the provided option or value.
        """
        if (
            self.vertical_text_displacement is None
            or self.vertical_text_displacement == "top"
        ):
            # Displace text upward by half the font size times a factor (e.g., 1.2)
            self.vertical_text_displacement = plt.rcParams["font.size"] / 2 * 1.2
        elif self.vertical_text_displacement == "bottom":
            # Displace text downward by half the font size times a factor
            self.vertical_text_displacement = -1 * plt.rcParams["font.size"] / 2 * 1.2
        else:
            # Use the provided displacement value
            self.vertical_text_displacement = self.vertical_text_displacement

    def inches_to_fig_fraction(self, pos):
        """
        Converts a position from inches to figure fraction coordinates.

        Args:
            pos (tuple of float): Position in inches as (x, y).

        Returns:
            numpy.ndarray: Position in figure fraction coordinates as (x, y).
        """
        # Convert position from inches to display coordinates (pixels)
        inch_pos = self.fig.dpi_scale_trans.transform(pos)
        # Convert display coordinates to figure fraction (0 to 1)
        fig_fraction_pos = self.fig.transFigure.inverted().transform(inch_pos)
        return fig_fraction_pos

    def draw(self):
        """
        Draws the arrow and places the text on the figure.

        Returns:
            tuple: A tuple containing the arrow and text artist objects, or the arrow if no text.
        """
        # Draw the arrow
        arrow = self.draw_arrow()

        # Place the text if it exists
        text_artist = self.place_text() if self.text else None

        return (arrow, text_artist) if text_artist else arrow

    def draw_arrow(self):
        """
        Draws the arrow annotation on the figure with an optional halo (outline) around the arrow.
        The halo properties are extracted from the self.halo dictionary if provided.

        Returns:
            matplotlib.text.Annotation: The arrow annotation artist object.
        """

        # Default halo settings
        halo_defaults = {
            "enabled": False,  # Whether to enable the halo or not
            "color": "white",  # Default halo color
            "scale": 3,  # Default scale of the halo relative to the arrow's linewidth
        }

        # Merge the provided halo settings with the defaults
        halo_settings = {**halo_defaults, **getattr(self, "halo", {})}

        # Check if units are in inches and convert positions if necessary
        if self.units == "inches":
            self.arrow_start_inches = self.inches_to_fig_fraction(self.start_pos)
            self.arrow_end_inches = self.inches_to_fig_fraction(self.end_pos)
        else:
            self.arrow_start_inches = self.start_pos
            self.arrow_end_inches = self.end_pos

        # If a halo is enabled, draw a thick arrow behind the actual arrow
        if halo_settings["enabled"]:

            halo_arrowprops = self.arrowprops.copy()
            halo_arrowprops["color"] = halo_settings["color"]  # Set the halo color
            # Scale the linewidth of the halo based on the original linewidth or lw
            halo_linewidth = self.arrowprops.get(
                "linewidth", self.arrowprops.get("lw", 2)
            )
            halo_arrowprops["linewidth"] = halo_linewidth * halo_settings["scale"]
            halo_arrowprops["lw"] = halo_linewidth * halo_settings["scale"]

            # Draw the halo (outline)
            self.ax.annotate(
                "",  # No text in the annotation itself
                xy=self.arrow_end_inches,  # End position of the arrow
                xycoords=self.scale,
                xytext=self.arrow_start_inches,  # Start position of the arrow
                arrowprops=halo_arrowprops,  # Arrow properties for the halo
            )

        # Draw the actual arrow on top
        arrow = self.ax.annotate(
            "",  # No text in the annotation itself
            xy=self.arrow_end_inches,  # End position of the arrow
            xycoords=self.scale,
            xytext=self.arrow_start_inches,  # Start position of the arrow
            arrowprops=self.arrowprops,  # Arrow properties for the main arrow
        )

        return arrow

    def place_text(self):
        """
        Places the text on the figure at the specified position relative to the arrow.

        Args:
            **textprops: Additional text properties for customizing the text appearance.
        """

        stroke_width = self.halo.get("font_stroke_width", None)
        stroke_color = self.halo.get("font_stroke_color", "black")
        
        

        # If textprops is not provided, initialize it as an empty dictionary
        if self.textprops is None:
            self.textprops = {}

        # Set the text color to match the arrow color if no text color is specified
        if "color" not in self.textprops:
            text_color = self.arrowprops.get(
                "color", "black"
            )  # Default to black if no color is in arrowprops
            self.textprops["color"] = text_color

        # Get the base position for the text
        text_x, text_y, ha, va = self._get_text_position()

        # Calculate the angle of the arrow to align the text
        angle = self.extract_angle()

        # Get the perpendicular vector to the arrow direction
        perpendicular_vector = get_perpendicular_vector(self.start_pos, self.end_pos)

        if self.units == "inches":
            # Shift the text position along the perpendicular vector
            shifted_position = shift_object_in_inches(
                self.fig,
                (text_x, text_y),
                perpendicular_vector,
                self.vertical_text_displacement,
            )

            # Place the text on the figure at the shifted position
            place_text_in_inches(
                self.fig,
                self.text,
                shifted_position[0],
                shifted_position[1],
                angle,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                **self.textprops,
            )

        elif self.units == "points":
            if self._ax is not None:
                obj = self.ax
            else:
                obj = self.fig

            # Shift the text position along the perpendicular vector
            shifted_position = shift_object_in_points(
                obj,
                (text_x, text_y),
                perpendicular_vector,
                self.vertical_text_displacement,
            )

            # Place the text on the figure at the shifted position
            place_text_points(
                self.fig,
                self.text,
                shifted_position[0],
                shifted_position[1],
                angle,
                self._ax,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                **self.textprops,
            )

    def _get_text_position(self):
        """
        Calculates the base position for the text based on the specified text position option.

        Returns:
            tuple: A tuple containing (text_x, text_y, horizontal_alignment, vertical_alignment).
        """
        if self.text_position == "center":
            # Position the text at the midpoint of the arrow
            text_x = (self.start_pos[0] + self.end_pos[0]) / 2
            text_y = (self.start_pos[1] + self.end_pos[1]) / 2
            ha = "center"
            va = "center"
        elif self.text_position == "start":
            # Position the text at the start of the arrow
            text_x = self.start_pos[0]
            text_y = self.start_pos[1]
            ha = "center"
            va = "center"
        elif self.text_position == "end":
            # Position the text at the end of the arrow
            text_x = self.end_pos[0]
            text_y = self.end_pos[1]
            ha = "center"
            va = "center"
        else:
            # Raise an error if an invalid text position is specified
            raise ValueError(
                f"Invalid text position: {self.text_position}, valid options are 'center', 'start', 'end'"
            )

        return text_x, text_y, ha, va

    def get_dx_dy(self):
        """
        Calculates the difference in x and y coordinates between the start and end positions.

        Returns:
            tuple of float: The differences in x and y coordinates as (dx, dy).
        """
        # Calculate the differences in x and y
        self.dx = self.end_pos[0] - self.start_pos[0]
        self.dy = self.end_pos[1] - self.start_pos[1]
        return self.dx, self.dy

    def extract_angle(self):
        """
        Calculates the angle of the arrow in degrees.

        Returns:
            float: The angle of the arrow in degrees.
        """
        # Get the differences in x and y coordinates
        dx, dy = self.get_dx_dy()

        # Calculate the angle in degrees using arctangent of dy/dx
        angle = np.degrees(np.arctan2(dy, dx))

        return angle


def draw_extended_arrow_indicator(
    fig,
    x,
    y,
    direction="vertical",
    text=None,
    offset=(-0.4, 0),
    offset_units="fraction",
    ax=None,
    vertical_text_displacement=None,
    text_position="center",
    text_alignment="center",
    units="points",
    arrowprops={},
    line_style={},
    halo=None,
    **annotation_kwargs,
):
    point_0 = obj_offset(
        (x[0], y[0]),  # position
        offset=offset,
        offset_units=offset_units,
        ax=ax,
    )
    point_1 = obj_offset(
        (x[1], y[1]),  # position
        offset=offset,
        offset_units=offset_units,
        ax=ax,
    )
    
    arrow = DrawArrow(
        fig,
        point_0,
        point_1,
        text=text,
        ax=ax,
        text_position=text_position,
        text_alignment=text_alignment,
        vertical_text_displacement=vertical_text_displacement,
        units=units,
        scale=annotation_kwargs.get("xycoords", "data"),
        arrowprops=arrowprops,
        halo=halo,
    )

    arrow.draw()

    if direction == "vertical":
        draw_lines(ax, [x[0], point_0[0]], [y[0], y[0]], style=line_style, halo=halo)
        draw_lines(ax, [x[0], point_1[0]], [y[1], y[1]], style=line_style, halo=halo)
    elif direction == "horizontal":
        draw_lines(ax, [x[0], x[0]], [y[0], point_0[1]], style=line_style, halo=halo)
        draw_lines(ax, [x[1], x[1]], [y[0], point_1[1]], style=line_style, halo=halo)
