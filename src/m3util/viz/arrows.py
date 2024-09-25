import matplotlib.pyplot as plt
import numpy as np


# class DrawArrow:
#     def __init__(
#         self,
#         fig,
#         start_pos,
#         end_pos,
#         text=None,
#         text_position="top",
#         text_alignment="center",
#         vertical_text_displacement=0,
#     ):
#         self.fig = fig
#         self.start_pos = start_pos
#         self.end_pos = end_pos
#         self.text = text
#         self.text_position = text_position
#         self.text_alignment = text_alignment
#         self.vertical_text_displacement = vertical_text_displacement

#     def draw(self):
#         # Draw the arrow
#         arrow = self.draw_arrow()

#         # Place the text if it exists
#         text_artist = self.place_text() if self.text else None
        
#         # Redraw the figure to reflect the updates
#         self.fig.canvas.draw()

#         return (arrow, text_artist) if text_artist else arrow

#     def draw_arrow(self, **arrowprops):
#         # Convert from inches to figure fraction
#         self.start_frac_inches = self.inches_to_fig_fraction(self.start_pos)
#         self.end_frac_inches = self.inches_to_fig_fraction(self.end_pos)

#         # Create an annotation with an arrow
#         arrow = plt.annotate(
#             "",
#             xy=self.end_frac_inches,
#             xycoords="figure fraction",
#             xytext=self.start_frac_inches,
#             textcoords="figure fraction",
#             arrowprops=dict(arrowstyle="->", **arrowprops),
#         )

#         return arrow

#     def extract_angle(self):
#         dx, dy = self.get_dx_dy()
#         angle = np.degrees(np.arctan2(dy, dx))
#         return angle

#     def get_dx_dy(self):
#         # Calculate the angle of the arrow
#         self.dx = self.end_pos[0] - self.start_pos[0]
#         self.dy = self.end_pos[1] - self.start_pos[1]
#         return self.dx, self.dy

#     def _get_perpendicular_vector(self):
#         # Calculate the perpendicular vector (choose one direction)
#         perp_vector = np.array([-self.dy, self.dx])

#         # Normalize the perpendicular vector
#         norm_perp_vector = perp_vector / np.linalg.norm(perp_vector)
#         return norm_perp_vector

#     def shift_perpendicular(self, position):
#         norm_perp_vector = self._get_perpendicular_vector()
#         # Calculate the shift in terms of the given distance
#         shift_vector = norm_perp_vector * self.vertical_text_displacement
#         # Calculate the new position
#         new_position = np.array(position) + shift_vector
#         return tuple(new_position)

#     def place_text(self, **textprops):
#         # Calculate the angle of the arrow
#         angle = self.extract_angle()

#         # get the text position
#         text_x, text_y, ha, va = self._get_text_position()

#         # Shift the text perpendicular to the arrow
#         position = self.shift_perpendicular((text_x, text_y))

#         # Create text at the specified position in figure fraction coordinates
#         text_artist = plt.text(
#             position[0],
#             position[1],
#             self.text,
#             rotation=angle,
#             horizontalalignment=ha,
#             verticalalignment=va,
#             transform=self.fig.transFigure,
#             **textprops,
#         )

#         return text_artist

#     def _get_text_position(self):
#         # Set text position and alignment
#         if self.text_position == "top":
#             text_x = (self.start_frac_inches[0] + self.end_frac_inches[0]) / 2
#             text_y = (self.start_frac_inches[1] + self.end_frac_inches[1]) / 2 + 0.02  # small offset
#             ha = self.text_alignment
#             va = "bottom"
#         elif self.text_position == "bottom":
#             text_x = (self.start_frac_inches[0] + self.end_frac_inches[0]) / 2
#             text_y = (self.start_frac_inches[1] + self.end_frac_inches[1]) / 2 - 0.02  # small offset
#             ha = self.text_alignment
#             va = "top"
#         elif self.text_position == "left":
#             text_x = self.start_frac_inches[0]
#             text_y = self.start_frac_inches[1]
#             ha = "right"
#             va = "center"
#         elif self.text_position == "right":
#             text_x = self.end_frac_inches[0]
#             text_y = self.end_frac_inches[1]
#             ha = "left"
#             va = "center"

#         return text_x, text_y, ha, va




import matplotlib.pyplot as plt
import numpy as np


class DrawArrow:
    def __init__(
        self,
        fig,
        start_pos,
        end_pos,
        text=None,
        text_position="top",
        text_alignment="center",
        vertical_text_displacement=0,
    ):
        self.fig = fig
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.text = text
        self.text_position = text_position
        self.text_alignment = text_alignment
        self.vertical_text_displacement = vertical_text_displacement
        
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
        self.start_frac_inches = self.inches_to_fig_fraction(self.start_pos)
        self.end_frac_inches = self.inches_to_fig_fraction(self.end_pos)

        # Create an annotation with an arrow
        arrow = plt.annotate(
            "",
            xy=self.end_frac_inches,
            xycoords="figure fraction",
            xytext=self.start_frac_inches,
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->", **arrowprops),
        )

        return arrow

    def extract_angle(self):

        dx, dy = self.get_dx_dy()

        angle = np.degrees(np.arctan2(dy, dx))

        return angle

    def get_dx_dy(self):

        # Calculate the angle of the arrow
        self.dx = self.end_pos[0] - self.start_pos[0]
        self.dy = self.end_pos[1] - self.start_pos[1]

        return self.dx, self.dy

    def _get_perpendicular_vector(self):
        # Calculate the perpendicular vector (choose one direction)
        perp_vector = np.array([-self.dy, self.dx])

        # Normalize the perpendicular vector
        norm_perp_vector = perp_vector / np.linalg.norm(perp_vector)

        return norm_perp_vector

    def shift_perpendicular(self, position):
        norm_perp_vector = self._get_perpendicular_vector()

        # Calculate the shift in terms of the given distance
        shift_vector = norm_perp_vector * self.vertical_text_displacement

        # Calculate the new position
        new_position = position + shift_vector

        return tuple(new_position)

    def place_text(self, **textprops):
        # Calculate the angle of the arrow
        angle = self.extract_angle()

        # get the text position
        text_x, text_y, ha, va = self._get_text_position()

        # Shift the text perpendicular to the arrow
        position = self.shift_perpendicular((text_x, text_y))

        # Convert position from inches to display coordinates
        disp_coords = self.fig.dpi_scale_trans.transform(position)

        # Create text at the specified position with initial settings
        text_artist = plt.text(
            disp_coords[0],
            disp_coords[1],
            self.text,
            rotation=angle,
            horizontalalignment="center",
            verticalalignment="center",
            transform=None,
            **textprops,
        )

        # Draw the figure to update renderer
        self.fig.canvas.draw()

        # Get the bounding box of the text in display coordinates
        bbox = text_artist.get_window_extent()

        # Calculate the adjustment needed to recenter the text
        # Since the rotation can displace the center, we adjust it back to the desired center
        bbox_center = np.array([bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2])
        shift = disp_coords - bbox_center

        # Update the text position with the calculated shift
        text_artist.set_x(disp_coords[0] + shift[0])
        text_artist.set_y(disp_coords[1] + shift[1])

        # Redraw the text in the new position
        self.fig.canvas.draw()

        return text_artist

    def _get_text_position(self):
        # Set text position and alignment
        if self.text_position == "top":
            text_x = (self.start_frac_inches[0] + self.end_frac_inches[0]) / 2
            text_y = (self.start_frac_inches[1] + self.end_frac_inches[1]) / 2 + 0.02  # small offset
            ha = self.text_alignment
            va = "bottom"
        elif self.text_position == "bottom":
            text_x = (self.start_frac_inches[0] + self.end_frac_inches[0]) / 2
            text_y = (self.start_frac_inches[1] + self.end_frac_inches[1]) / 2 - 0.02  # small offset
            ha = self.text_alignment
            va = "top"
        elif self.text_position == "left":
            text_x = self.start_frac_inches[0]
            text_y = self.start_frac_inches[1]
            ha = "right"
            va = "center"
        elif self.text_position == "right":
            text_x = self.end_frac_inches[0]
            text_y = self.end_frac_inches[1]
            ha = "left"
            va = "center"

        return text_x, text_y, ha, va
