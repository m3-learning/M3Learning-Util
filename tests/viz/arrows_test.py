import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.text import Annotation
import matplotlib.patheffects as path_effects
from matplotlib.text import Text


# Assuming draw_ellipse_with_arrow is in a module named 'plot_utils'
from m3util.viz.arrows import draw_ellipse_with_arrow, place_text_in_inches


####### Draw ellipse with arrow tests #######

def test_valid_input_horizontal():
    """Test the function with valid inputs for horizontal line direction."""
    fig, ax = plt.subplots()
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)
    
    draw_ellipse_with_arrow(
        ax=ax,
        x_data=x_data,
        y_data=y_data,
        value=5.0,
        width=0.1,
        height=0.05,
        line_direction='horizontal'
    )
    
    # Check if an ellipse has been added to the axis
    ellipses = [patch for patch in ax.patches if isinstance(patch, Ellipse)]
    assert len(ellipses) == 1, "Ellipse was not added to the plot"
    
    # Check if an annotation (arrow) has been added to the axis
    annotations = ax.texts
    assert len(annotations) == 1, "Arrow annotation was not added to the plot"
    
    # Check if the arrow annotation has the expected properties
    arrow = annotations[0]
    assert arrow.arrowprops is not None, "Arrow properties not found in annotation"


def test_valid_input_vertical():
    """Test the function with valid inputs for vertical line direction."""
    fig, ax = plt.subplots()
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)
    
    draw_ellipse_with_arrow(
        ax=ax,
        x_data=x_data,
        y_data=y_data,
        value=5.0,
        width=0.1,
        height=0.05,
        line_direction='vertical'
    )
    
    # Check if an ellipse has been added to the axis
    ellipses = [patch for patch in ax.patches if isinstance(patch, Ellipse)]
    assert len(ellipses) == 1, "Ellipse was not added to the plot"
    
    # Check if an annotation (arrow) has been added to the axis
    annotations = ax.texts
    assert len(annotations) == 1, "Arrow annotation was not added to the plot"
    
    # Check if the arrow annotation has the expected properties
    arrow = annotations[0]
    assert arrow.arrowprops is not None, "Arrow properties not found in annotation"

def test_invalid_line_direction():
    """Test if ValueError is raised for an invalid line direction."""
    fig, ax = plt.subplots()
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)
    
    with pytest.raises(ValueError, match="line_direction must be 'horizontal' or 'vertical'"):
        draw_ellipse_with_arrow(
            ax=ax,
            x_data=x_data,
            y_data=y_data,
            value=5.0,
            width=0.1,
            height=0.05,
            line_direction='diagonal'
        )

def test_invalid_arrow_position():
    """Test if ValueError is raised for an invalid arrow position."""
    fig, ax = plt.subplots()
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data)
    
    with pytest.raises(ValueError, match="arrow_position must be 'top' or 'bottom'"):
        draw_ellipse_with_arrow(
            ax=ax,
            x_data=x_data,
            y_data=y_data,
            value=5.0,
            width=0.1,
            height=0.05,
            arrow_position='left'
        )

def test_invalid_x_y_data_shape():
    """Test if ValueError is raised when x_data and y_data have different shapes."""
    fig, ax = plt.subplots()
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data[:-1])  # Make y_data have one less element
    
    with pytest.raises(ValueError, match="x_data and y_data must have the same shape."):
        draw_ellipse_with_arrow(
            ax=ax,
            x_data=x_data,
            y_data=y_data,
            value=5.0,
            width=0.1,
            height=0.05
        )

####### Place Text in Inches tests #######

def test_basic_text_placement():
    """Test that the text is placed correctly at the specified position."""
    fig = plt.figure()
    text_str = "Test Text"
    
    # Call the function to place text at (2, 3) inches with no stroke
    text_artist = place_text_in_inches(fig, text_str, 2, 3, angle=0)
    
    assert isinstance(text_artist, Text), "The returned object is not a matplotlib.text.Text instance"
    assert text_artist.get_text() == text_str, "The text content is incorrect"
    
    # Verify the text is centered and at the right position in figure inches
    # Since this is figure dependent, we ensure the text coordinates match in display coordinates
    display_coords = fig.dpi_scale_trans.transform((2, 3))
    assert text_artist.get_position() == (display_coords[0], display_coords[1]), "Text position is incorrect"
    assert text_artist.get_rotation() == 0, "Text rotation angle is incorrect"

def test_text_with_rotation():
    """Test that the text is placed and rotated correctly."""
    fig = plt.figure()
    
    # Place text with a 45 degree rotation
    text_artist = place_text_in_inches(fig, "Rotated Text", 2, 3, angle=45)
    
    # Check that the rotation angle is correctly set
    assert text_artist.get_rotation() == 45, "The rotation angle of the text is incorrect"

def test_text_with_stroke():
    """Test that the text stroke is applied correctly."""
    fig = plt.figure()
    
    # Call the function with stroke enabled
    text_artist = place_text_in_inches(
        fig, 
        "Stroke Text", 
        2, 
        3, 
        angle=0, 
        stroke_width=2, 
        stroke_color="red", 
        fontsize=12, 
        color="blue"
    )
    
    # Force a draw of the figure to ensure path effects are applied
    fig.canvas.draw()

    # Check that stroke is applied correctly
    path_effects_list = text_artist.get_path_effects()
    assert len(path_effects_list) == 2, "There should be two path effects: Stroke and Normal"
    
    # The first path effect should be a stroke
    stroke_effect = path_effects_list[0]
    assert isinstance(stroke_effect, path_effects.Stroke), "First path effect is not a Stroke"


    
def test_text_without_stroke():
    """Test that no stroke is applied when stroke_width is None."""
    fig = plt.figure()
    
    # Place text without stroke
    text_artist = place_text_in_inches(fig, "No Stroke", 2, 3, angle=0, fontsize=12, color="green")
    
    # Check that no path effects are applied
    path_effects_list = text_artist.get_path_effects()
    assert len(path_effects_list) == 0, "There should be no path effects applied"

def test_text_properties():
    """Test that additional text properties are applied correctly."""
    fig = plt.figure()
    
    # Place text with additional properties (fontsize and color)
    text_artist = place_text_in_inches(
        fig, 
        "Custom Text", 
        2, 
        3, 
        angle=0, 
        fontsize=20, 
        color="purple"
    )
    
    # Check that the text properties are set correctly
    assert text_artist.get_fontsize() == 20, "The fontsize is incorrect"
    assert text_artist.get_color() == "purple", "The text color is incorrect"
