import matplotlib.pyplot as plt
import numpy as np


def place_text_in_inches(fig, text, x_inch, y_inch, angle, **textprops):
    # Convert from inches to display coordinates using the figure's dpi scale transform
    display_coords = fig.dpi_scale_trans.transform((x_inch, y_inch))

    # Place the text using the calculated display coordinates
    text_artist = plt.text(
        display_coords[0],
        display_coords[1],
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=None,  # No additional transformation since we use display coordinates
        rotation=angle,
        **textprops,
    )

    # Trigger the figure redraw
    fig.canvas.draw()

    return text_artist


def shift_object_in_inches(fig, position_inch, direction_vector, n_points):
    """
    Shifts an object by n points along a specified vector direction, and returns the new position in inches.

    :param fig: The matplotlib figure to get the DPI for point-to-inch conversion
    :param position_inch: The starting position in inches (tuple of x, y in inches)
    :param direction_vector: The direction vector for the shift (tuple of dx, dy)
    :param n_points: The number of points to shift along the direction vector
    :return: The new position in inches (tuple of x, y in inches)
    """
    # Normalize the direction vector
    direction_vector = np.array(direction_vector)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Get the figure's DPI to convert points to inches
    dpi = fig.dpi
    points_per_inch = 72  # 1 inch = 72 points

    # Convert the shift in points to inches
    shift_inch = n_points / points_per_inch

    # Calculate the shift vector in inches
    shift_vector_inch = direction_vector * shift_inch

    # Apply the shift to the original position (which is already in inches)
    new_position_inch = np.array(position_inch) + shift_vector_inch

    return tuple(new_position_inch)


def get_perpendicular_vector(point1, point2, clockwise=False):
    """
    Computes the perpendicular vector from two endpoints.

    :param point1: The first endpoint as a tuple (x1, y1)
    :param point2: The second endpoint as a tuple (x2, y2)
    :param clockwise: If True, returns the clockwise perpendicular vector.
                      Otherwise, returns the counterclockwise perpendicular vector.
    :return: The perpendicular vector as a tuple (dx, dy)
    """
    # Calculate the direction vector
    x1, y1 = point1
    x2, y2 = point2
    direction_vector = np.array([x2 - x1, y2 - y1])

    # Compute the perpendicular vector
    if clockwise:
        perpendicular_vector = np.array(
            [direction_vector[1], -direction_vector[0]]
        )  # Clockwise
    else:
        perpendicular_vector = np.array(
            [-direction_vector[1], direction_vector[0]]
        )  # Counterclockwise

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
        self.fig = fig
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.text = text
        self.text_position = text_position
        self.text_alignment = text_alignment
        self.vertical_text_displacement = vertical_text_displacement
        self.set_vertical_text_displacement()

    def set_vertical_text_displacement(self):
        if (
            self.vertical_text_displacement is None
            or self.vertical_text_displacement == "top"
        ):
            self.vertical_text_displacement = plt.rcParams["font.size"]/2 * 1.2
        elif self.vertical_text_diplacement == "bottom":
            self.vertical_text_displacement = -1 * plt.rcParams["font.size"]/2 * 1.2
        else:
            self.vertical_text_displacement = self.vertical_text_displacement

    def inches_to_fig_fraction(self, pos):
        """
        Convert position from inches to figure fraction.
        :param pos: Position in inches (tuple of x, y in inches)
        :return: Position in figure fraction (x, y in fraction)
        """
        inch_pos = self.fig.dpi_scale_trans.transform(pos)
        fig_fraction_pos = self.fig.transFigure.inverted().transform(inch_pos)
        return fig_fraction_pos

    def draw(self):
        # Draw the arrow
        arrow = self.draw_arrow()

        # Place the text if it exists
        text_artist = self.place_text() if self.text else None

        return (arrow, text_artist) if text_artist else arrow

    def draw_arrow(self, **arrowprops):
        # Convert inches to figure fraction
        self.arrow_start_inches = self.inches_to_fig_fraction(self.start_pos)
        self.arrow_end_inches = self.inches_to_fig_fraction(self.end_pos)

        # Create an annotation with an arrow
        arrow = plt.annotate(
            "",
            xy=self.arrow_end_inches,
            xycoords="figure fraction",
            xytext=self.arrow_start_inches,
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->", **arrowprops),
        )

        return arrow

    def place_text(self, **textprops):
        # get the position
        text_x, text_y, ha, va = self._get_text_position()

        # Calculate the angle of the arrow
        angle = self.extract_angle()

        perpendicular_vector = get_perpendicular_vector(self.start_pos, self.end_pos)

        shifted_position = shift_object_in_inches(
            self.fig,
            (text_x, text_y),
            perpendicular_vector,
            self.vertical_text_displacement,
        )

        place_text_in_inches(
            self.fig,
            self.text,
            shifted_position[0],
            shifted_position[1],
            angle,
            **textprops,
        )

    def _get_text_position(self):
        if self.text_position == "center":
            text_x = (self.start_pos[0] + self.end_pos[0]) / 2
            text_y = (self.start_pos[1] + self.end_pos[1]) / 2
            ha = "center"
            va = "center"
        elif self.text_position == "start":
            text_x = self.start_pos[0]
            text_y = self.start_pos[1]
            ha = "center"
            va = "center"
        elif self.text_position == "end":
            text_x = self.end_pos[0]
            text_y = self.end_pos[1]
            ha = "center"
            va = "center"
        else:
            raise ValueError(
                f"Invalid text position: {self.text_position}, valid options are 'center', 'start', 'end'"
            )

        return text_x, text_y, ha, va

    def get_dx_dy(self):
        # Calculate the angle of the arrow
        self.dx = self.end_pos[0] - self.start_pos[0]
        self.dy = self.end_pos[1] - self.start_pos[1]
        return self.dx, self.dy

    def extract_angle(self):
        dx, dy = self.get_dx_dy()

        angle = np.degrees(np.arctan2(dy, dx))

        return angle

    # def shift_perpendicular(self, position):
    #     norm_perp_vector = self._get_perpendicular_vector()

    #     # Calculate the shift in terms of the given distance
    #     shift_vector = norm_perp_vector * self.vertical_text_displacement

    #     # Calculate the new position
    #     new_position = position + shift_vector

    #     return tuple(new_position)

    # def place_text(self, **textprops):
    #     # Calculate the angle of the arrow
    #     angle = self.extract_angle()

    #     # get the text position
    #     text_x, text_y, ha, va = self._get_text_position()

    #     position = (text_x, text_y)

    #     # # Shift the text perpendicular to the arrow
    #     # position = self.shift_perpendicular((text_x, text_y))

    #     # Create text at the specified position with initial settings
    #     text_artist = plt.text(
    #         position[0],
    #         position[1],
    #         self.text,
    #         rotation=angle,
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #         transform=None,
    #         **textprops,
    #     )

    #     # Draw the figure to update renderer
    #     self.fig.canvas.draw()

    #     # # Get the bounding box of the text in display coordinates
    #     # bbox = text_artist.get_window_extent()

    #     # # Calculate the adjustment needed to recenter the text
    #     # # Since the rotation can displace the center, we adjust it back to the desired center
    #     # bbox_center = np.array([bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2])
    #     # shift = position - bbox_center

    #     shift = (0, 0)

    #     # Update the text position with the calculated shift
    #     text_artist.set_x(position[0] + shift[0])
    #     text_artist.set_y(position[1] + shift[1])

    #     # Redraw the text in the new position
    #     self.fig.canvas.draw()

    #     return text_artist
