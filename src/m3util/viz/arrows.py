import matplotlib.pyplot as plt
import numpy as np


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
        text_position="center",
        text_alignment="center",
        vertical_text_displacement=None,
    ):
        """
        Initializes the DrawArrow object with arrow and text properties.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure to draw on.
            start_pos (tuple of float): The starting position of the arrow in inches (x, y).
            end_pos (tuple of float): The ending position of the arrow in inches (x, y).
            text (str, optional): The text to display alongside the arrow. Defaults to None.
            text_position (str, optional): Position of the text relative to the arrow.
                Options are 'center', 'start', or 'end'. Defaults to 'center'.
            text_alignment (str, optional): Alignment of the text. Defaults to 'center'.
            vertical_text_displacement (float or str, optional): Vertical displacement of the text in points.
                If None or 'top', displaces the text upward by half the font size.
                If 'bottom', displaces the text downward by half the font size.
                If a float is provided, uses that value as the displacement.
                Defaults to None.
        """
        # Initialize object properties
        self.fig = fig
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.text = text
        self.text_position = text_position
        self.text_alignment = text_alignment
        self.vertical_text_displacement = vertical_text_displacement
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
        arrow = plt.annotate(
            "",  # No text in the annotation itself
            xy=self.arrow_end_inches,  # End position of the arrow
            xycoords="figure fraction",
            xytext=self.arrow_start_inches,  # Start position of the arrow
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->", **arrowprops),  # Arrow properties
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
