import matplotlib.pyplot as plt
import numpy as np

def draw_arrow(fig, start_pos, end_pos, text=None, text_position='top', text_alignment='center', **arrowprops):
    """
    Draws an arrow on a given figure with positions specified in inches and adds optional text with alignment.
    
    Parameters:
    - fig: matplotlib Figure object
    - start_pos: Tuple (x, y) for the start position in inches
    - end_pos: Tuple (x, y) for the end position in inches
    - text: Optional text to display along with the arrow
    - text_position: 'top', 'bottom', 'left', 'right' to position text relative to the arrow
    - text_alignment: 'left', 'right', 'center' for text justification
    - arrowprops: Additional properties for the arrow (color, linewidth, etc.)
    """
    # Convert inches to figure fraction
    start_frac = fig.transFigure.inverted().transform(fig.dpi_scale_trans.transform(start_pos))
    end_frac = fig.transFigure.inverted().transform(fig.dpi_scale_trans.transform(end_pos))
    
    # Create an annotation with an arrow
    arrow = plt.annotate('', xy=end_frac, xycoords='figure fraction',
                         xytext=start_frac, textcoords='figure fraction',
                         arrowprops=dict(arrowstyle="->", **arrowprops))
    
    # Calculate the angle of the arrow
    dx = end_frac[0] - start_frac[0]
    dy = end_frac[1] - start_frac[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Set text position and alignment
    if text_position == 'top':
        text_x = (start_frac[0] + end_frac[0]) / 2
        text_y = (start_frac[1] + end_frac[1]) / 2 + 0.02  # small offset
        ha = text_alignment
        va = 'bottom'
    elif text_position == 'bottom':
        text_x = (start_frac[0] + end_frac[0]) / 2
        text_y = (start_frac[1] + end_frac[1]) / 2 - 0.02  # small offset
        ha = text_alignment
        va = 'top'
    elif text_position == 'left':
        text_x = start_frac[0]
        text_y = start_frac[1]
        ha = 'right'
        va = 'center'
    elif text_position == 'right':
        text_x = end_frac[0]
        text_y = end_frac[1]
        ha = 'left'
        va = 'center'
    
    # Add text if specified
    if text:
        plt.text(text_x, text_y, text, horizontalalignment=ha, verticalalignment=va,
                 rotation=angle, rotation_mode='anchor', transform=fig.transFigure)
    
    return arrow

def place_rotated_text(fig, position, text, angle, **textprops):
    """
    Places rotated text on a figure such that the specified position remains the center of the text.

    Parameters:
    - fig: matplotlib Figure object
    - position: Tuple (x, y) specifying the position in inches where the center of the text should be
    - text: String to be displayed
    - angle: Rotation angle in degrees
    - textprops: Additional properties for the text (color, fontsize, etc.)
    """
    # Convert position from inches to display coordinates
    disp_coords = fig.dpi_scale_trans.transform(position)

    # Create text at the specified position with initial settings
    text_artist = plt.text(disp_coords[0], disp_coords[1], text, rotation=angle,
                           horizontalalignment='center', verticalalignment='center',
                           transform=None, **textprops)

    # Draw the figure to update renderer
    fig.canvas.draw()

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
    fig.canvas.draw()

    return text_artist

def shift_perpendicular(position, direction_vector, distance):
    """
    Shifts a position by a specified distance along a direction perpendicular to a given vector.

    Parameters:
    - position: Tuple or list (x, y) representing the original position.
    - direction_vector: Tuple or list (dx, dy) representing the direction vector.
    - distance: Distance to shift the position in points.

    Returns:
    - Tuple (new_x, new_y) representing the new position.
    """
    # Extract components of the direction vector
    dx, dy = direction_vector
    
    # Calculate the perpendicular vector (choose one direction)
    perp_vector = np.array([-dy, dx])
    
    # Normalize the perpendicular vector
    norm_perp_vector = perp_vector / np.linalg.norm(perp_vector)
    
    # Calculate the shift in terms of the given distance
    shift_vector = norm_perp_vector * distance
    
    # Calculate the new position
    new_position = np.array(position) + shift_vector
    
    return tuple(new_position)