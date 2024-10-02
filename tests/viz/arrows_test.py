import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.text import Annotation



# Assuming draw_ellipse_with_arrow is in a module named 'plot_utils'
from m3util.viz.arrows import draw_ellipse_with_arrow


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
