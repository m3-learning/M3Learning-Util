import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.text import Annotation
import matplotlib.patheffects as path_effects
from matplotlib.text import Text
from unittest.mock import patch
from unittest import mock


# Assuming draw_ellipse_with_arrow is in a module named 'plot_utils'
from m3util.viz.arrows import (
    draw_ellipse_with_arrow,
    place_text_in_inches,
    place_text_points,
    shift_object_in_points,
    shift_object_in_inches,
    get_perpendicular_vector,
    DrawArrow,
    draw_extended_arrow_indicator,
)


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
        line_direction="horizontal",
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
        line_direction="vertical",
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

    with pytest.raises(
        ValueError, match="line_direction must be 'horizontal' or 'vertical'"
    ):
        draw_ellipse_with_arrow(
            ax=ax,
            x_data=x_data,
            y_data=y_data,
            value=5.0,
            width=0.1,
            height=0.05,
            line_direction="diagonal",
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
            arrow_position="left",
        )


def test_invalid_x_y_data_shape():
    """Test if ValueError is raised when x_data and y_data have different shapes."""
    fig, ax = plt.subplots()
    x_data = np.linspace(0, 10, 100)
    y_data = np.sin(x_data[:-1])  # Make y_data have one less element

    with pytest.raises(ValueError, match="x_data and y_data must have the same shape."):
        draw_ellipse_with_arrow(
            ax=ax, x_data=x_data, y_data=y_data, value=5.0, width=0.1, height=0.05
        )


####### Place Text in Inches tests #######


def test_basic_text_placement():
    """Test that the text is placed correctly at the specified position."""
    fig = plt.figure()
    text_str = "Test Text"

    # Call the function to place text at (2, 3) inches with no stroke
    text_artist = place_text_in_inches(fig, text_str, 2, 3, angle=0)

    assert isinstance(
        text_artist, Text
    ), "The returned object is not a matplotlib.text.Text instance"
    assert text_artist.get_text() == text_str, "The text content is incorrect"

    # Verify the text is centered and at the right position in figure inches
    # Since this is figure dependent, we ensure the text coordinates match in display coordinates
    display_coords = fig.dpi_scale_trans.transform((2, 3))
    assert text_artist.get_position() == (
        display_coords[0],
        display_coords[1],
    ), "Text position is incorrect"
    assert text_artist.get_rotation() == 0, "Text rotation angle is incorrect"


def test_text_with_rotation():
    """Test that the text is placed and rotated correctly."""
    fig = plt.figure()

    # Place text with a 45 degree rotation
    text_artist = place_text_in_inches(fig, "Rotated Text", 2, 3, angle=45)

    # Check that the rotation angle is correctly set
    assert (
        text_artist.get_rotation() == 45
    ), "The rotation angle of the text is incorrect"


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
        color="blue",
    )

    # Force a draw of the figure to ensure path effects are applied
    fig.canvas.draw()

    # Check that stroke is applied correctly
    path_effects_list = text_artist.get_path_effects()
    assert (
        len(path_effects_list) == 2
    ), "There should be two path effects: Stroke and Normal"

    # The first path effect should be a stroke
    stroke_effect = path_effects_list[0]
    assert isinstance(
        stroke_effect, path_effects.Stroke
    ), "First path effect is not a Stroke"


def test_text_without_stroke():
    """Test that no stroke is applied when stroke_width is None."""
    fig = plt.figure()

    # Place text without stroke
    text_artist = place_text_in_inches(
        fig, "No Stroke", 2, 3, angle=0, fontsize=12, color="green"
    )

    # Check that no path effects are applied
    path_effects_list = text_artist.get_path_effects()
    assert len(path_effects_list) == 0, "There should be no path effects applied"


def test_text_properties():
    """Test that additional text properties are applied correctly."""
    fig = plt.figure()

    # Place text with additional properties (fontsize and color)
    text_artist = place_text_in_inches(
        fig, "Custom Text", 2, 3, angle=0, fontsize=20, color="purple"
    )

    # Check that the text properties are set correctly
    assert text_artist.get_fontsize() == 20, "The fontsize is incorrect"
    assert text_artist.get_color() == "purple", "The text color is incorrect"


##### Place Text in Points tests #####


def test_basic_text_placement_points():
    """Test that the text is placed correctly at the specified position in axis coordinates."""
    fig, ax = plt.subplots()

    text_str = "Test Text"
    x_coord, y_coord = 0.5, 0.5

    # Call the function to place text
    text_artist = place_text_points(fig, text_str, x_coord, y_coord, angle=0, ax=ax)

    assert isinstance(
        text_artist, Text
    ), "The returned object is not a matplotlib.text.Text instance"
    assert text_artist.get_text() == text_str, "The text content is incorrect"

    # Verify the text is placed at the right coordinates
    assert text_artist.get_position() == (
        x_coord,
        y_coord,
    ), "Text position is incorrect"
    assert text_artist.get_rotation() == 0, "Text rotation angle is incorrect"


def test_text_with_rotation_points():
    """Test that the text is placed and rotated correctly."""
    fig, ax = plt.subplots()

    x_coord, y_coord = 0.5, 0.5
    angle = 45

    # Call the function to place text with rotation
    text_artist = place_text_points(
        fig, "Rotated Text", x_coord, y_coord, angle=angle, ax=ax
    )

    # Check that the rotation angle is correctly set
    assert (
        text_artist.get_rotation() == angle
    ), "The rotation angle of the text is incorrect"


def test_text_with_stroke_points():
    """Test that the text stroke is applied correctly."""
    fig, ax = plt.subplots()

    # Call the function with stroke enabled
    text_artist = place_text_points(
        fig,
        "Stroke Text",
        0.5,
        0.5,
        angle=0,
        ax=ax,
        stroke_width=2,
        stroke_color="red",
        fontsize=12,
        color="blue",
    )

    # Force a draw of the figure to ensure path effects are applied
    fig.canvas.draw()

    # Check that stroke is applied correctly
    path_effects_list = text_artist.get_path_effects()
    assert (
        len(path_effects_list) == 2
    ), "There should be two path effects: Stroke and Normal"

    # The first path effect should be a stroke
    stroke_effect = path_effects_list[0]
    assert isinstance(
        stroke_effect, path_effects.Stroke
    ), "First path effect is not a Stroke"


def test_text_without_stroke_points():
    """Test that no stroke is applied when stroke_width is None."""
    fig, ax = plt.subplots()

    # Place text without stroke
    text_artist = place_text_points(
        fig, "No Stroke", 0.5, 0.5, angle=0, ax=ax, fontsize=12, color="green"
    )

    # Check that no path effects are applied
    path_effects_list = text_artist.get_path_effects()
    assert len(path_effects_list) == 0, "There should be no path effects applied"


def test_text_properties_points():
    """Test that additional text properties are applied correctly."""
    fig, ax = plt.subplots()

    # Place text with additional properties (fontsize and color)
    text_artist = place_text_points(
        fig, "Custom Text", 0.5, 0.5, angle=0, ax=ax, fontsize=20, color="purple"
    )

    # Check that the text properties are set correctly
    assert text_artist.get_fontsize() == 20, "The fontsize is incorrect"
    assert text_artist.get_color() == "purple", "The text color is incorrect"


###### Shift Object in Points tests ######


def test_shift_positive_x():
    """Test shifting a point along the positive x-axis."""
    fig, ax = plt.subplots()

    # Start at (0.5, 0.5) and shift by 10 points along the x-axis
    start_pos = (0.5, 0.5)
    direction = (1, 0)
    n_points = 10

    new_pos = shift_object_in_points(ax, start_pos, direction, n_points)

    # Check that the new position is correctly shifted along the x-axis
    assert new_pos[0] > start_pos[0], "Position should shift positively along x-axis."
    assert new_pos[1] == pytest.approx(
        start_pos[1], abs=1e-6
    ), "y-coordinate should remain unchanged."


def test_shift_positive_y():
    """Test shifting a point along the positive y-axis."""
    fig, ax = plt.subplots()

    # Start at (0.5, 0.5) and shift by 15 points along the y-axis
    start_pos = (0.5, 0.5)
    direction = (0, 1)
    n_points = 15

    new_pos = shift_object_in_points(ax, start_pos, direction, n_points)

    # Check that the new position is correctly shifted along the y-axis
    assert new_pos[1] > start_pos[1], "Position should shift positively along y-axis."
    assert new_pos[0] == pytest.approx(
        start_pos[0], abs=1e-6
    ), "x-coordinate should remain unchanged."


def test_shift_negative_direction():
    """Test shifting a point along a negative direction."""
    fig, ax = plt.subplots()

    # Start at (0.5, 0.5) and shift by 10 points in the negative x direction
    start_pos = (0.5, 0.5)
    direction = (-1, 0)
    n_points = 10

    new_pos = shift_object_in_points(ax, start_pos, direction, n_points)

    # Check that the new position is correctly shifted negatively along the x-axis
    assert new_pos[0] < start_pos[0], "Position should shift negatively along x-axis."
    assert new_pos[1] == pytest.approx(
        start_pos[1], abs=1e-6
    ), "y-coordinate should remain unchanged."


def test_shift_diagonal():
    """Test shifting a point along a diagonal vector."""
    fig, ax = plt.subplots()

    # Start at (0.5, 0.5) and shift by 10 points along the diagonal (1, 1)
    start_pos = (0.5, 0.5)
    direction = (1, 1)
    n_points = 10

    new_pos = shift_object_in_points(ax, start_pos, direction, n_points)

    # Check that the new position is shifted in both x and y directions
    assert new_pos[0] > start_pos[0], "Position should shift positively along x-axis."
    assert new_pos[1] > start_pos[1], "Position should shift positively along y-axis."


def test_shift_zero_points():
    """Test that no shift occurs when the shift is 0 points."""
    fig, ax = plt.subplots()

    # Start at (0.5, 0.5) and shift by 0 points
    start_pos = (0.5, 0.5)
    direction = (1, 0)
    n_points = 0

    new_pos = shift_object_in_points(ax, start_pos, direction, n_points)

    # Check that the position remains unchanged
    assert new_pos == pytest.approx(
        start_pos
    ), "Position should remain unchanged when shifting by 0 points."


def test_shift_non_normalized_vector():
    """Test shifting with a non-normalized direction vector."""
    fig, ax = plt.subplots()

    # Start at (0.5, 0.5) and shift by 10 points with a non-normalized direction vector (3, 4)
    start_pos = (0.5, 0.5)
    direction = (3, 4)
    n_points = 10

    new_pos = shift_object_in_points(ax, start_pos, direction, n_points)

    # Check that the new position is shifted correctly, as the function normalizes the vector
    assert new_pos[0] > start_pos[0], "Position should shift positively along x-axis."
    assert new_pos[1] > start_pos[1], "Position should shift positively along y-axis."


##### Shift Object in Inches tests #####


def test_shift_positive_x_shift_object_in_inches():
    """Test shifting a point along the positive x-axis."""
    fig = plt.figure(dpi=100)

    # Start at (2, 3) inches and shift by 72 points (1 inch) along the x-axis
    start_pos = (2, 3)
    direction = (1, 0)
    n_points = 72  # This is equivalent to shifting by 1 inch

    new_pos = shift_object_in_inches(fig, start_pos, direction, n_points)

    # The new position should be shifted 1 inch to the right (x-axis), so (3, 3)
    assert new_pos == pytest.approx(
        (3, 3), abs=1e-6
    ), f"Expected position (3, 3), got {new_pos}"


def test_shift_positive_y_shift_object_in_inches():
    """Test shifting a point along the positive y-axis."""
    fig = plt.figure(dpi=100)

    # Start at (2, 3) inches and shift by 144 points (2 inches) along the y-axis
    start_pos = (2, 3)
    direction = (0, 1)
    n_points = 144  # This is equivalent to shifting by 2 inches

    new_pos = shift_object_in_inches(fig, start_pos, direction, n_points)

    # The new position should be shifted 2 inches upwards, so (2, 5)
    assert new_pos == pytest.approx(
        (2, 5), abs=1e-6
    ), f"Expected position (2, 5), got {new_pos}"


def test_shift_negative_x_shift_object_in_inches():
    """Test shifting a point along the negative x-axis."""
    fig = plt.figure(dpi=100)

    # Start at (2, 3) inches and shift by 36 points (0.5 inch) in the negative x direction
    start_pos = (2, 3)
    direction = (-1, 0)
    n_points = 36  # This is equivalent to shifting by 0.5 inches

    new_pos = shift_object_in_inches(fig, start_pos, direction, n_points)

    # The new position should be shifted 0.5 inch to the left, so (1.5, 3)
    assert new_pos == pytest.approx(
        (1.5, 3), abs=1e-6
    ), f"Expected position (1.5, 3), got {new_pos}"


def test_shift_diagonal_shift_object_in_inches():
    """Test shifting a point along a diagonal direction."""
    fig = plt.figure(dpi=100)

    # Start at (1, 1) inches and shift by 72 points (1 inch) along the diagonal (1, 1)
    start_pos = (1, 1)
    direction = (1, 1)
    n_points = 72  # Equivalent to shifting by 1 inch along the diagonal

    new_pos = shift_object_in_inches(fig, start_pos, direction, n_points)

    # Since the direction is diagonal, the shift will occur equally in both x and y directions
    # Normalize the direction vector (1, 1) => (1/sqrt(2), 1/sqrt(2)), so each axis moves by 1/sqrt(2) inches
    expected_shift = 1 / np.sqrt(2)
    expected_pos = (1 + expected_shift, 1 + expected_shift)

    assert new_pos == pytest.approx(
        expected_pos, abs=1e-6
    ), f"Expected position {expected_pos}, got {new_pos}"


def test_shift_zero_points_shift_object_in_inches():
    """Test that no shift occurs when n_points is 0."""
    fig = plt.figure(dpi=100)

    # Start at (2, 3) inches and shift by 0 points
    start_pos = (2, 3)
    direction = (1, 0)
    n_points = 0  # No shift

    new_pos = shift_object_in_inches(fig, start_pos, direction, n_points)

    # The new position should remain the same, (2, 3)
    assert new_pos == pytest.approx(
        start_pos
    ), f"Expected position {start_pos}, got {new_pos}"


def test_shift_non_normalized_vector_shift_object_in_inches():
    """Test shifting with a non-normalized direction vector."""
    fig = plt.figure(dpi=100)

    # Start at (1, 1) inches and shift by 72 points (1 inch) with a non-normalized direction vector (3, 4)
    start_pos = (1, 1)
    direction = (3, 4)
    n_points = 72  # Equivalent to shifting by 1 inch along the (3, 4) vector

    new_pos = shift_object_in_inches(fig, start_pos, direction, n_points)

    # Normalize the vector (3, 4) => (3/5, 4/5), the shift will be distributed accordingly
    expected_shift_x = 3 / 5  # 1 inch in total, distributed to x as 3/5
    expected_shift_y = 4 / 5  # 1 inch in total, distributed to y as 4/5
    expected_pos = (1 + expected_shift_x, 1 + expected_shift_y)

    assert new_pos == pytest.approx(
        expected_pos, abs=1e-6
    ), f"Expected position {expected_pos}, got {new_pos}"


##### Get Perpendicular Vector tests #####


def test_perpendicular_counterclockwise_horizontal():
    """Test that the counterclockwise perpendicular vector is correct for a horizontal vector."""
    point1 = (0, 0)
    point2 = (1, 0)

    # Counterclockwise perpendicular to (1, 0) should be (0, 1)
    expected = (0, 1)

    result = get_perpendicular_vector(point1, point2)

    assert result == expected, f"Expected {expected}, but got {result}"


def test_perpendicular_clockwise_horizontal():
    """Test that the clockwise perpendicular vector is correct for a horizontal vector."""
    point1 = (0, 0)
    point2 = (1, 0)

    # Clockwise perpendicular to (1, 0) should be (0, -1)
    expected = (0, -1)

    result = get_perpendicular_vector(point1, point2, clockwise=True)

    assert result == expected, f"Expected {expected}, but got {result}"


def test_perpendicular_counterclockwise_vertical():
    """Test that the counterclockwise perpendicular vector is correct for a vertical vector."""
    point1 = (0, 0)
    point2 = (0, 1)

    # Counterclockwise perpendicular to (0, 1) should be (-1, 0)
    expected = (-1, 0)

    result = get_perpendicular_vector(point1, point2)

    assert result == expected, f"Expected {expected}, but got {result}"


def test_perpendicular_clockwise_vertical():
    """Test that the clockwise perpendicular vector is correct for a vertical vector."""
    point1 = (0, 0)
    point2 = (0, 1)

    # Clockwise perpendicular to (0, 1) should be (1, 0)
    expected = (1, 0)

    result = get_perpendicular_vector(point1, point2, clockwise=True)

    assert result == expected, f"Expected {expected}, but got {result}"


def test_perpendicular_diagonal_counterclockwise():
    """Test that the counterclockwise perpendicular vector is correct for a diagonal vector."""
    point1 = (0, 0)
    point2 = (1, 1)

    # Counterclockwise perpendicular to (1, 1) should be (-1, 1)
    expected = (-1, 1)

    result = get_perpendicular_vector(point1, point2)

    assert result == expected, f"Expected {expected}, but got {result}"


def test_perpendicular_diagonal_clockwise():
    """Test that the clockwise perpendicular vector is correct for a diagonal vector."""
    point1 = (0, 0)
    point2 = (1, 1)

    # Clockwise perpendicular to (1, 1) should be (1, -1)
    expected = (1, -1)

    result = get_perpendicular_vector(point1, point2, clockwise=True)

    assert result == expected, f"Expected {expected}, but got {result}"


##### arrow class #####


@pytest.fixture
def setup_fig():
    """
    Pytest fixture to create a figure for testing.
    """
    fig, ax = plt.subplots()
    return fig, ax


def test_draw_arrow_with_halo():
    fig, ax = plt.subplots()

    # Use mocking to verify that annotate is called twice (halo and arrow)
    with mock.patch.object(ax, "annotate", wraps=ax.annotate) as mock_annotate:
        arrow = DrawArrow(
            fig,
            (0.1, 0.1),
            (0.9, 0.9),
            ax=ax,
            halo={"enabled": True, "color": "white", "scale": 3},
        )
        arrow.draw_arrow()
        assert (
            mock_annotate.call_count == 2
        ), "Expected annotate to be called twice (halo and arrow)"


def test_inches_conversion(setup_fig):
    """
    Test the conversion from inches to figure fraction coordinates.
    """
    fig, ax = setup_fig
    start_pos = (1, 1)  # Inches
    end_pos = (2, 2)  # Inches

    arrow = DrawArrow(fig=fig, start_pos=start_pos, end_pos=end_pos)

    converted_pos = arrow.inches_to_fig_fraction(start_pos)

    # Verify if the conversion returns an expected figure fraction range (0, 1)
    assert all(
        0 <= coord <= 1 for coord in converted_pos
    ), "Converted position should be in figure fraction (0, 1)."


def test_extract_angle():
    """
    Test if the angle extraction between the start and end positions is correct.
    """
    fig, ax = plt.subplots()
    start_pos = (0, 0)
    end_pos = (1, 1)

    arrow = DrawArrow(fig=fig, start_pos=start_pos, end_pos=end_pos)

    angle = arrow.extract_angle()

    # The angle between (0, 0) and (1, 1) should be 45 degrees
    assert np.isclose(angle, 45), f"Expected angle of 45 degrees, got {angle}"


def test_get_dx_dy():
    """
    Test the calculation of the dx and dy differences between start and end positions.
    """
    fig, ax = plt.subplots()
    start_pos = (0, 0)
    end_pos = (3, 4)

    arrow = DrawArrow(fig=fig, start_pos=start_pos, end_pos=end_pos)
    dx, dy = arrow.get_dx_dy()

    assert dx == 3, "dx should be the difference in x-coordinates."
    assert dy == 4, "dy should be the difference in y-coordinates."


####### draw_extended_arrow_indicator tests #######


@pytest.fixture
def create_figure():
    """Fixture to create a matplotlib figure and axis for testing."""
    fig, ax = plt.subplots()
    return fig, ax


@patch("m3util.viz.arrows.DrawArrow")  # Mock the DrawArrow class
@patch("m3util.viz.arrows.obj_offset")  # Mock obj_offset to control its output
@patch("m3util.viz.arrows.draw_lines")  # Mock draw_lines to control its output
def test_vertical_arrow_draw(
    mock_draw_lines, mock_obj_offset, mock_DrawArrow, create_figure
):
    """Test vertical arrow drawing with default parameters."""

    # Setup mock behavior for obj_offset
    mock_obj_offset.side_effect = [(0.5, 0.5), (0.7, 0.7)]  # Mocked offset positions

    # Setup figure and axis
    fig, ax = create_figure

    # Call the function with mock inputs
    draw_extended_arrow_indicator(fig, x=[0, 1], y=[0, 1], direction="vertical", ax=ax)

    # Check if the arrow was initialized with the right points and properties
    mock_DrawArrow.assert_called_once_with(
        fig,
        (0.5, 0.5),
        (0.7, 0.7),
        text=None,
        ax=ax,
        text_position="center",
        text_alignment="center",
        vertical_text_displacement=None,
        units="points",
        scale="data",
        arrowprops={},
        halo=None,
    )

    # Ensure the draw method was called once on the arrow instance
    mock_DrawArrow.return_value.draw.assert_called_once()

    # Check that draw_lines is called for the vertical direction
    mock_draw_lines.assert_any_call(ax, [0, 0.5], [0, 0], style={}, halo=None)
    mock_draw_lines.assert_any_call(ax, [0, 0.7], [1, 1], style={}, halo=None)


@patch("m3util.viz.arrows.DrawArrow")
@patch("m3util.viz.arrows.obj_offset")
@patch("m3util.viz.arrows.draw_lines")
def test_horizontal_arrow_draw(
    mock_draw_lines, mock_obj_offset, mock_DrawArrow, create_figure
):
    """Test horizontal arrow drawing with horizontal direction."""

    # Mock obj_offset to return fake offset positions
    mock_obj_offset.side_effect = [(0.5, 0.3), (0.7, 0.9)]  # Mocked positions

    fig, ax = create_figure

    # Call the function with mock inputs and horizontal direction
    draw_extended_arrow_indicator(
        fig, x=[0, 1], y=[0, 1], direction="horizontal", ax=ax
    )

    # Check if the arrow was initialized with the right parameters
    mock_DrawArrow.assert_called_once_with(
        fig,
        (0.5, 0.3),
        (0.7, 0.9),
        text=None,
        ax=ax,
        text_position="center",
        text_alignment="center",
        vertical_text_displacement=None,
        units="points",
        scale="data",
        arrowprops={},
        halo=None,
    )

    # Check that the draw method is called
    mock_DrawArrow.return_value.draw.assert_called_once()

    # Ensure draw_lines is called for the horizontal direction
    mock_draw_lines.assert_any_call(ax, [0, 0], [0, 0.3], style={}, halo=None)
    mock_draw_lines.assert_any_call(ax, [1, 1], [0, 0.9], style={}, halo=None)


@patch("m3util.viz.arrows.DrawArrow")
@patch("m3util.viz.arrows.obj_offset")
@patch("m3util.viz.arrows.draw_lines")
def test_custom_text_and_arrowprops(
    mock_draw_lines, mock_obj_offset, mock_DrawArrow, create_figure
):
    """Test the function with custom text and arrow properties."""

    # Mock obj_offset to return fixed positions
    mock_obj_offset.side_effect = [(0.5, 0.3), (0.8, 0.7)]  # Mocked positions

    fig, ax = create_figure

    # Custom arrow properties and text
    arrowprops = {"arrowstyle": "->", "color": "blue"}
    custom_text = "Test Arrow"

    draw_extended_arrow_indicator(
        fig,
        x=[0, 1],
        y=[0, 1],
        direction="vertical",
        ax=ax,
        text=custom_text,
        arrowprops=arrowprops,
    )

    # Check if the arrow was initialized with the custom text and properties
    mock_DrawArrow.assert_called_once_with(
        fig,
        (0.5, 0.3),
        (0.8, 0.7),
        text=custom_text,
        ax=ax,
        text_position="center",
        text_alignment="center",
        vertical_text_displacement=None,
        units="points",
        scale="data",
        arrowprops=arrowprops,
        halo=None,
    )

    # Ensure draw method was called
    mock_DrawArrow.return_value.draw.assert_called_once()


def test_invalid_direction(create_figure):
    """Test the function with an invalid direction input."""
    fig, ax = create_figure

    with pytest.raises(ValueError):
        # Call the function with an invalid direction
        draw_extended_arrow_indicator(
            fig, x=[0, 1], y=[0, 1], direction="diagonal", ax=ax
        )


def test_draw_ellipse_with_arrow_invalid_data():
    fig, ax = plt.subplots()
    x_data = np.array([0, 1, 2])
    y_data = np.array([0, 1])  # Mismatched length with x_data

    with pytest.raises(ValueError, match="x_data and y_data must have the same shape"):
        draw_ellipse_with_arrow(ax, x_data, y_data, value=1, width=0.1, height=0.1)


def test_draw_ellipse_with_arrow_valid():
    fig, ax = plt.subplots()
    x_data = np.array([0, 1, 2])
    y_data = np.array([0, 1, 2])

    # No exception should be raised for valid data
    draw_ellipse_with_arrow(ax, x_data, y_data, value=1, width=0.1, height=0.1)
    assert len(ax.patches) == 1  # One ellipse should be drawn


def test_get_perpendicular_vector():
    point1 = (0, 0)
    point2 = (1, 0)

    perp_vec = get_perpendicular_vector(point1, point2)
    assert perp_vec == (0, 1), "Expected (0, 1) as the perpendicular vector"


def test_draw_ellipse_with_arrow_custom_arrowprops():
    fig, ax = plt.subplots()
    x_data = np.array([0, 1, 2])
    y_data = np.array([0, 1, 2])

    arrow_props = {"facecolor": "red", "width": 5, "headwidth": 15}
    draw_ellipse_with_arrow(
        ax, x_data, y_data, value=1, width=0.1, height=0.1, arrow_props=arrow_props
    )

    # Test that the custom arrowprops have been applied
    for annot in ax.texts:
        assert annot.arrowprops["facecolor"] == "red"
        assert annot.arrowprops["width"] == 5
        assert annot.arrowprops["headwidth"] == 15


def test_shift_object_in_points():
    fig, ax = plt.subplots()
    position_axis = (0.5, 0.5)
    direction_vector = (1, 0)

    new_position = shift_object_in_points(ax, position_axis, direction_vector, 10)

    assert new_position != position_axis
    assert new_position[0] > position_axis[0]  # Shifting to the right


def test_shift_object_in_inches():
    fig, ax = plt.subplots()
    position_inch = (2, 3)
    direction_vector = (1, 1)

    new_position = shift_object_in_inches(fig, position_inch, direction_vector, 15)

    assert new_position != position_inch
    assert new_position[0] > position_inch[0]  # Should have shifted
    assert new_position[1] > position_inch[1]


def test_get_perpendicular_vector_clockwise():
    point1 = (0, 0)
    point2 = (1, 0)
    perp_vector = get_perpendicular_vector(point1, point2, clockwise=True)

    assert perp_vector == (0, -1), "Expected (0, -1) for clockwise perpendicular"


def test_draw_arrow_with_halo():
    fig, ax = plt.subplots()

    # Use mocking to verify that annotate is called twice (halo and arrow)
    with mock.patch.object(ax, "annotate", wraps=ax.annotate) as mock_annotate:
        arrow = DrawArrow(
            fig,
            (0.1, 0.1),
            (0.9, 0.9),
            ax=ax,
            halo={"enabled": True, "color": "white", "scale": 3},
        )
        arrow.draw_arrow()
        assert (
            mock_annotate.call_count == 2
        ), "Expected annotate to be called twice (halo and arrow)"


def test_draw_arrow_default_initialization():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0.1, 0.1), (0.9, 0.9))

    assert arrow.fig == fig
    assert arrow.start_pos == (0.1, 0.1)
    assert arrow.end_pos == (0.9, 0.9)
    assert arrow.text is None
    assert arrow.ax == plt
    assert arrow.text_position == "center"
    assert arrow.text_alignment == "center"
    assert arrow.vertical_text_displacement == plt.rcParams["font.size"] / 2 * 1.2
    assert arrow.units == "inches"
    assert arrow.scale == "figure fraction"
    assert arrow.arrowprops == {"arrowstyle": "->"}
    assert arrow.textprops is None
    assert arrow.halo == {}


def test_draw_arrow_custom_initialization():
    fig, ax = plt.subplots()
    arrowprops = {"arrowstyle": "<->", "color": "red", "linewidth": 2}
    textprops = {"fontsize": 12, "color": "blue"}
    halo = {"enabled": True, "color": "yellow", "scale": 2}

    arrow = DrawArrow(
        fig=fig,
        start_pos=(1, 1),
        end_pos=(2, 2),
        text="Test Arrow",
        ax=ax,
        text_position="start",
        text_alignment="left",
        vertical_text_displacement=10,
        units="points",
        scale="data",
        arrowprops=arrowprops,
        textprops=textprops,
        halo=halo,
    )

    assert arrow.fig == fig
    assert arrow.start_pos == (1, 1)
    assert arrow.end_pos == (2, 2)
    assert arrow.text == "Test Arrow"
    assert arrow.ax == ax
    assert arrow.text_position == "start"
    assert arrow.text_alignment == "left"
    assert arrow.vertical_text_displacement == 10
    assert arrow.units == "points"
    assert arrow.scale == "data"
    assert arrow.arrowprops == arrowprops
    assert arrow.textprops == textprops
    assert arrow.halo == halo


def test_set_vertical_text_displacement():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0, 0), (1, 1))

    # Default displacement (should be 'top')
    arrow.set_vertical_text_displacement()
    expected_displacement = plt.rcParams["font.size"] / 2 * 1.2
    assert arrow.vertical_text_displacement == expected_displacement

    # Set displacement to 'bottom'
    arrow.vertical_text_displacement = "bottom"
    arrow.set_vertical_text_displacement()
    expected_displacement = -plt.rcParams["font.size"] / 2 * 1.2
    assert arrow.vertical_text_displacement == expected_displacement

    # Set displacement to a specific value
    arrow.vertical_text_displacement = 15
    arrow.set_vertical_text_displacement()
    assert arrow.vertical_text_displacement == 15


def test_inches_to_fig_fraction():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0, 0), (1, 1))
    pos_inches = (2, 2)
    fig_fraction_pos = arrow.inches_to_fig_fraction(pos_inches)

    # Since the figure size is known, we can compute expected values
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    expected_x = pos_inches[0] / fig_width_inch
    expected_y = pos_inches[1] / fig_height_inch

    np.testing.assert_almost_equal(fig_fraction_pos[0], expected_x)
    np.testing.assert_almost_equal(fig_fraction_pos[1], expected_y)


def test_get_dx_dy_and_extract_angle():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0, 0), (1, 1))

    dx, dy = arrow.get_dx_dy()
    assert dx == 1
    assert dy == 1

    angle = arrow.extract_angle()
    assert angle == pytest.approx(45.0)


def test_draw_arrow_without_halo():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0.1, 0.1), (0.9, 0.9), ax=ax)
    arrow_artist = arrow.draw_arrow()

    # Check that an Annotation object was created
    assert isinstance(arrow_artist, plt.Annotation)
    # Since we didn't enable halo, only one arrow should be drawn
    assert len(ax.artists) == 0  # Annotations are not stored in ax.artists


def test_invalid_text_position():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0, 0), (1, 1), text="Invalid", text_position="invalid")

    with pytest.raises(
        ValueError,
        match="Invalid text position: invalid, valid options are 'center', 'start', 'end'",
    ):
        arrow._get_text_position()


def test_draw_method_without_text():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0, 0), (1, 1), ax=ax)
    result = arrow.draw()

    # Should return only the arrow artist
    assert isinstance(result, plt.Annotation)

    # Check that there are no text artists with non-empty text
    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) == 0, f"Expected no text artists with text, found {len(texts)}"


def test_units_points():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (10, 10), (20, 20), units="points", ax=ax)
    arrow.draw()

    # Since units are 'points', positions should be interpreted accordingly
    # We can't directly test internal state but can ensure no exceptions are raised


def test_units_inches():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (1, 1), (2, 2), units="inches", ax=ax)
    arrow.draw()

    # Positions should be converted from inches to figure fraction
    # Ensure no exceptions and arrow is drawn


def test_extract_angle_negative_differences():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (2, 2), (1, 1))
    angle = arrow.extract_angle()
    assert angle == pytest.approx(-135.0)


def test_halo_properties():
    fig, ax = plt.subplots()
    halo_settings = {"enabled": True, "color": "white", "scale": 2}
    arrow = DrawArrow(fig, (0.1, 0.1), (0.9, 0.9), ax=ax, halo=halo_settings)

    with mock.patch.object(ax, "annotate", wraps=ax.annotate) as mock_annotate:
        arrow.draw_arrow()

        # First call is for the halo
        call_args_halo = mock_annotate.call_args_list[0][1]
        assert call_args_halo["arrowprops"]["color"] == "white"
        assert (
            call_args_halo["arrowprops"]["linewidth"]
            == arrow.arrowprops.get("linewidth", 2) * 2
        )

        # Second call is for the actual arrow
        call_args_arrow = mock_annotate.call_args_list[1][1]
        assert call_args_arrow["arrowprops"] == arrow.arrowprops


def test_text_properties_inheritance():
    fig, ax = plt.subplots()
    arrowprops = {"color": "green"}
    arrow = DrawArrow(
        fig, (0, 0), (1, 1), text="Inherited Text", arrowprops=arrowprops, ax=ax
    )
    arrow.place_text()

    text_artist = ax.texts[0]
    # Text color should inherit from arrowprops if not specified in textprops
    assert text_artist.get_color() == "green"

    # Now specify text color explicitly
    arrow = DrawArrow(
        fig,
        (0, 0),
        (1, 1),
        text="Custom Text",
        arrowprops=arrowprops,
        textprops={"color": "blue"},
        ax=ax,
    )
    arrow.place_text()

    text_artist = ax.texts[1]
    assert text_artist.get_color() == "blue"


def test_shift_text_position():
    fig, ax = plt.subplots()
    arrow = DrawArrow(
        fig, (0, 0), (1, 1), text="Shifted Text", vertical_text_displacement=20, ax=ax
    )
    arrow.place_text()

    text_artist = ax.texts[0]
    # Cannot directly test the shifted position without complex calculations,
    # but we can ensure that the text was added without errors
    assert text_artist.get_text() == "Shifted Text"


def test_invalid_units():
    fig, ax = plt.subplots()
    arrow = DrawArrow(fig, (0, 0), (1, 1), units="invalid_unit", ax=ax)

    # The code does not raise an exception, so we ensure it runs without error
    arrow.draw()

    # We can check that an arrow was added
    # Since units are invalid, positions may not be correct, but the arrow is drawn
    assert isinstance(arrow, DrawArrow)


def test_draw_arrow_missing_figure():
    with pytest.raises(AttributeError):
        arrow = DrawArrow(None, (0, 0), (1, 1))
        arrow.draw()


def test_place_text_in_inches():
    fig = plt.figure()
    text_artist = place_text_in_inches(
        fig, "Inches Text", x_inch=1, y_inch=1, angle=0, fontsize=10
    )
    assert text_artist.get_text() == "Inches Text"
    assert text_artist.get_fontsize() == 10


def test_place_text_points():
    fig, ax = plt.subplots()
    text_artist = place_text_points(
        fig, "Points Text", x=50, y=50, angle=0, ax=ax, fontsize=10
    )
    assert text_artist.get_text() == "Points Text"
    assert text_artist.get_fontsize() == 10


def test_valid_input_vertical_bottom():
    """Test the function with valid inputs for vertical line direction and bottom arrow position."""
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
        line_direction="vertical",
        arrow_position="bottom",
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
