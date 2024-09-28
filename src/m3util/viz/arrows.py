import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from m3util.viz.layout import get_closest_point

def draw_ellipse_with_arrow(ax, x_data, y_data, value, width, height, axis='x',
                                    line_direction='horizontal', arrow_position='top',
                                    arrow_length_frac=0.3, color='blue', linewidth=2,
                                    arrow_props=None, ellipse_props=None, arrow_direction = 'positive'):
    """
    Draw an ellipse with an arrow at a specific point on a line plot.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis where the ellipse and arrow will be drawn.
        x_data (array-like): X data points of the line plot.
        y_data (array-like): Y data points of the line plot.
        value (float): The x or y value at which to place the ellipse.
        width (float): Width of the ellipse as a fraction of the x-axis range.
        height (float): Height of the ellipse as a fraction of the y-axis range.
        axis (str, optional): Axis to find the closest point on (default is 'x').
        line_direction (str, optional): Direction of the line to which the ellipse and arrow are related (default is 'horizontal').
        arrow_position (str, optional): Position to place the arrow relative to the ellipse (default is 'top').
        arrow_length_frac (float, optional): Length of the arrow as a fraction of the axis range (default is 0.3).
        color (str, optional): Color of the ellipse and arrow (default is 'blue').
        linewidth (float, optional): Line width of the ellipse (default is 2).
        arrow_props (dict, optional): Additional properties to customize the arrow appearance.
        ellipse_props (dict, optional): Additional properties to customize the ellipse appearance.
        arrow_direction (str, optional): Direction of the arrow (default is 'positive').

    Raises:
        ValueError: If invalid values are provided for axis, line_direction, or arrow_position.
    """
    # Ensure x_data and y_data are NumPy arrays
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    # Check that x_data and y_data have the same length
    if x_data.shape != y_data.shape:
        raise ValueError("x_data and y_data must have the same shape.")

    # Get the closest point on the line plot
    ellipse_center = get_closest_point(x_data, y_data, value, axis=axis)

    # Get axis limits for scaling dimensions and arrow length
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Calculate arrow length based on line direction
    if line_direction == 'horizontal':
        arrow_length = arrow_length_frac * (x_max - x_min)
        if arrow_direction == 'negative':
            arrow_length = -arrow_length
    elif line_direction == 'vertical':
        arrow_length = arrow_length_frac * (y_max - y_min)
        if arrow_direction == 'negative':
            arrow_length = -arrow_length
    else:
        raise ValueError("line_direction must be 'horizontal' or 'vertical'")

    # Scale the width and height of the ellipse
    width_scaled = width * (x_max - x_min)
    height_scaled = height * (y_max - y_min)

    # Set default properties for the ellipse and update with any additional properties
    default_ellipse_props = {'edgecolor': color, 'facecolor': 'none', 'lw': linewidth}
    if ellipse_props:
        default_ellipse_props.update(ellipse_props)

    # Draw the ellipse
    ellipse = Ellipse(xy=ellipse_center, width=width_scaled, height=height_scaled, **default_ellipse_props)
    ax.add_patch(ellipse)

    # Calculate the start and end points of the arrow based on position and direction
    if line_direction == 'horizontal':
        if arrow_position == 'top':
            start_point = (ellipse_center[0], ellipse_center[1] + height_scaled / 2)
            end_point = (ellipse_center[0] + arrow_length, ellipse_center[1] + height_scaled / 2)
        elif arrow_position == 'bottom':
            start_point = (ellipse_center[0], ellipse_center[1] - height_scaled / 2)
            end_point = (ellipse_center[0] + arrow_length, ellipse_center[1] - height_scaled / 2)
        else:
            raise ValueError("arrow_position must be 'top' or 'bottom'")
    elif line_direction == 'vertical':
        if arrow_position == 'top':
            start_point = (ellipse_center[0] + width_scaled / 2, ellipse_center[1])
            end_point = (ellipse_center[0] + width_scaled / 2, ellipse_center[1] + arrow_length)
        elif arrow_position == 'bottom':
            start_point = (ellipse_center[0] - width_scaled / 2, ellipse_center[1])
            end_point = (ellipse_center[0] - width_scaled / 2, ellipse_center[1] + arrow_length)
        else:
            raise ValueError("arrow_position must be 'top' or 'bottom'")
    else:
        raise ValueError("line_direction must be 'horizontal' or 'vertical'")

    # Set default properties for the arrow and update with any additional properties
    default_arrow_props = {'facecolor': color, 'width': 2, 'headwidth': 10, 'headlength': 10, 'linewidth': 0}
    if arrow_props:
        default_arrow_props.update(arrow_props)

    # Draw the arrow
    ax.annotate('', xy=end_point, xytext=start_point, arrowprops=default_arrow_props)

def place_text_in_inches(fig, text, x_inch, y_inch, angle, **textprops):
    """
    Places text on a matplotlib figure at a specified position in inches.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure on which to place the text.
        text (str): The text string to be displayed.
        x_inch (float): The x-coordinate in inches from the left of the figure.
        y_inch (float): The y-coordinate in inches from the bottom of the figure.
        angle (float): The rotation angle of the text in degrees.
        **textprops: Additional keyword arguments for text properties (e.g., fontsize, color).

    Returns:
        matplotlib.text.Text: The text artist object added to the figure.
    """
    # Convert from inches to display coordinates (pixels) using the figure's dpi scale transform
    display_coords = fig.dpi_scale_trans.transform((x_inch, y_inch))

    # Place the text using the calculated display coordinates
    text_artist = plt.text(
        display_coords[0],  # x-coordinate in display (pixel) coordinates
        display_coords[1],  # y-coordinate in display (pixel) coordinates
        text,  # Text string to display
        horizontalalignment="center",  # Horizontal alignment of the text
        verticalalignment="center",  # Vertical alignment of the text
        transform=None,  # No additional transformation since we use display coordinates
        rotation=angle,  # Rotation angle of the text
        **textprops,  # Additional text properties
    )

    # Trigger the figure redraw to update the display with the new text
    fig.canvas.draw()

    return text_artist


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
        ax = None,
        text_position="center",
        text_alignment="center",
        vertical_text_displacement=None,
        units="inches",
        scale="figure fraction",
        arrow_props = dict(arrowstyle="->")
    ):
        
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
        self.arrow_props = arrow_props
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

    def draw_arrow(self, **arrowprops):
        """
        Draws the arrow annotation on the figure.

        Args:
            **arrowprops: Additional properties for customizing the arrow appearance.

        Returns:
            matplotlib.text.Annotation: The arrow annotation artist object.
        """
        # Convert start and end positions from inches to figure fraction coordinates
        self.arrow_start_inches = self.inches_to_fig_fraction(self.start_pos)
        self.arrow_end_inches = self.inches_to_fig_fraction(self.end_pos)
        
        # Create an annotation with an arrow between the start and end positions
        arrow = self.ax.annotate(
            "",  # No text in the annotation itself
            xy=self.arrow_end_inches,  # End position of the arrow
            xycoords=self.scale,
            xytext=self.arrow_start_inches,  # Start position of the arrow
            textcoords=self.scale,
            arrowprops=self.arrow_props,  # Arrow properties
        )

        return arrow

    def place_text(self, **textprops):
        """
        Places the text on the figure at the specified position relative to the arrow.

        Args:
            **textprops: Additional text properties for customizing the text appearance.
        """
        # Get the base position for the text
        text_x, text_y, ha, va = self._get_text_position()

        # Calculate the angle of the arrow to align the text
        angle = self.extract_angle()

        # Get the perpendicular vector to the arrow direction
        perpendicular_vector = get_perpendicular_vector(self.start_pos, self.end_pos)

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
            **textprops,
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
