import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def draw_arrow(fig, start_pos, end_pos, text=None, text_position='top', **arrowprops):
    """
    Draws an arrow on a given figure with positions specified in inches and adds optional text.
    
    Parameters:
    - fig: matplotlib Figure object
    - start_pos: Tuple (x, y) for the start position in inches
    - end_pos: Tuple (x, y) for the end position in inches
    - text: Optional text to display along with the arrow
    - text_position: 'top' or 'bottom' to position text above or below the arrow
    - arrowprops: Additional properties for the arrow (color, linewidth, etc.)
    """
    # Convert inches to figure fraction
    start_frac = fig.transFigure.inverted().transform(fig.dpi_scale_trans.transform(start_pos))
    end_frac = fig.transFigure.inverted().transform(fig.dpi_scale_trans.transform(end_pos))
    
    # Determine the mid-point for placing text
    mid_x = (start_frac[0] + end_frac[0]) / 2
    mid_y = (start_frac[1] + end_frac[1]) / 2
    
    # Adjust text alignment based on its position relative to the arrow
    vertical_alignment = 'bottom' if text_position == 'top' else 'top'
    vertical_offset = 0.02 if text_position == 'top' else -0.02  # Offset to avoid overlapping with the arrow

    # Create an annotation with an arrow
    arrow = plt.annotate('', xy=end_frac, xycoords='figure fraction',
                         xytext=start_frac, textcoords='figure fraction',
                         arrowprops=dict(arrowstyle="->", **arrowprops))
    
    # Add text if specified
    if text:
        plt.text(mid_x, mid_y + vertical_offset, text, horizontalalignment='center',
                 verticalalignment=vertical_alignment, transform=fig.transFigure)
    
    return arrow
