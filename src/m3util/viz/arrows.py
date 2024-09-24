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