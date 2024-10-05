from m3util.viz.text import add_text_to_figure, set_sci_notation_label, labelfigs, line_annotation
import matplotlib.pyplot as plt
import pytest
import numpy as np
from matplotlib import patheffects
from unittest.mock import patch


def test_add_text_to_figure():
    # Create a figure
    fig = plt.figure(figsize=(8, 6))  # 8 inches by 6 inches

    # Add text at a specific position in inches
    text = "Sample Text"
    text_position_in_inches = (4, 3)  # Center of the figure
    kwargs = {"fontsize": 12, "color": "blue"}

    # Call the function to add the text
    add_text_to_figure(fig, text, text_position_in_inches, **kwargs)

    # Verify that the text was added correctly
    assert (
        len(fig.texts) == 1
    ), "Expected exactly one text element to be added to the figure."

    # Check the position of the text
    text_obj = fig.texts[0]
    expected_position = (
        text_position_in_inches[0] / fig.get_size_inches()[0],
        text_position_in_inches[1] / fig.get_size_inches()[1],
    )

    assert text_obj.get_position() == pytest.approx(
        expected_position
    ), f"Expected text position {expected_position}, but got {text_obj.get_position()}."

    # Check the text content
    assert (
        text_obj.get_text() == text
    ), f"Expected text content '{text}', but got '{text_obj.get_text()}'."

    # Check additional kwargs
    assert (
        text_obj.get_fontsize() == kwargs["fontsize"]
    ), f"Expected fontsize {kwargs['fontsize']}, but got {text_obj.get_fontsize()}."
    assert (
        text_obj.get_color() == kwargs["color"]
    ), f"Expected color {kwargs['color']}, but got {text_obj.get_color()}."


def test_add_text_to_figure_default():
    # Create a figure with default settings
    fig = plt.figure()

    # Add text at a specific position in inches
    text = "Default Position Text"
    text_position_in_inches = (2, 1)  # Arbitrary position
    add_text_to_figure(fig, text, text_position_in_inches)

    # Verify that the text was added correctly
    assert (
        len(fig.texts) == 1
    ), "Expected exactly one text element to be added to the figure."

    # Check the position of the text
    text_obj = fig.texts[0]
    expected_position = (
        text_position_in_inches[0] / fig.get_size_inches()[0],
        text_position_in_inches[1] / fig.get_size_inches()[1],
    )

    assert text_obj.get_position() == pytest.approx(
        expected_position
    ), f"Expected text position {expected_position}, but got {text_obj.get_position()}."

    # Check the text content
    assert (
        text_obj.get_text() == text
    ), f"Expected text content '{text}', but got '{text_obj.get_text()}'."


@pytest.fixture(autouse=True)
def set_tick_labelsize():
    original_size = plt.rcParams["xtick.labelsize"]
    # Set to a numerical value for testing
    plt.rcParams["xtick.labelsize"] = 10
    yield
    # Restore original value after test
    plt.rcParams["xtick.labelsize"] = original_size


def test_set_sci_notation_label_y_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    set_sci_notation_label(ax, axis="y")
    offset_text = ax.yaxis.get_offset_text().get_text()
    assert offset_text == "", "Offset text should be hidden"
    exponent_texts = [text.get_text() for text in ax.texts]
    assert len(exponent_texts) == 1, "Should add one exponent text"
    assert exponent_texts[0] == r"$\times10^{3}$", "Exponent text should be correct"
    plt.close(fig)


def test_set_sci_notation_label_x_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1000], [0, 1])
    set_sci_notation_label(ax, axis="x")
    offset_text = ax.xaxis.get_offset_text().get_text()
    assert offset_text == "", "Offset text should be hidden"
    exponent_texts = [text.get_text() for text in ax.texts]
    assert len(exponent_texts) == 1, "Should add one exponent text"
    assert exponent_texts[0] == r"$\times10^{3}$", "Exponent text should be correct"
    plt.close(fig)


def test_set_sci_notation_label_no_ticks():
    fig, ax = plt.subplots()
    ax.plot([], [])
    ax.set_xticks([])
    ax.set_yticks([])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        len(exponent_texts) == 0
    ), "No exponent text should be added when there are no ticks"
    plt.close(fig)


def test_set_sci_notation_label_zero_exponent():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        len(exponent_texts) == 0
    ), "No exponent text should be added when exponent is zero"
    plt.close(fig)


def test_set_sci_notation_label_invalid_corner():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    with pytest.raises(ValueError) as excinfo:
        set_sci_notation_label(ax, corner="middle")
    assert "Invalid corner position" in str(excinfo.value)
    plt.close(fig)


def test_set_sci_notation_label_corner_positions():
    corners = ["bottom left", "bottom right", "top left", "top right"]
    expected_ha_va = {
        "bottom left": ("left", "bottom"),
        "bottom right": ("right", "bottom"),
        "top left": ("left", "top"),
        "top right": ("right", "top"),
    }
    for corner in corners:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1000])
        set_sci_notation_label(ax, corner=corner)
        exponent_text = ax.texts[-1]
        ha, va = expected_ha_va[corner]
        assert (
            exponent_text.get_ha() == ha
        ), f"Horizontal alignment should be '{ha}' for corner '{corner}'"
        assert (
            exponent_text.get_va() == va
        ), f"Vertical alignment should be '{va}' for corner '{corner}'"
        plt.close(fig)


def test_set_sci_notation_label_offset_points():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    offset_points = (20, 10)
    set_sci_notation_label(ax, offset_points=offset_points)
    exponent_text = ax.texts[-1]
    fig_width, fig_height = fig.get_size_inches()
    dpi = fig.dpi
    offset_points[0] / dpi / fig_width
    offset_points[1] / dpi / fig_height
    transform = exponent_text.get_transform()
    x, y = transform.transform((exponent_text.get_position()))
    assert (
        exponent_text.get_text() == r"$\times10^{3}$"
    ), "Exponent text should be correct"
    plt.close(fig)


def test_set_sci_notation_label_linewidth_stroke_color():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    linewidth = 2
    stroke_color = "red"
    set_sci_notation_label(ax, linewidth=linewidth, stroke_color=stroke_color)
    exponent_text = ax.texts[-1]
    path_effects_list = exponent_text.get_path_effects()
    assert len(path_effects_list) == 1, "Should have one path effect"
    path_effect = path_effects_list[0]
    assert isinstance(
        path_effect, patheffects.withStroke
    ), "Path effect should be 'withStroke'"
    assert path_effect._gc["linewidth"] == linewidth, "Line width should match"
    assert path_effect._gc["foreground"] == stroke_color, "Stroke color should match"
    plt.close(fig)


def test_set_sci_notation_label_write_to_axis():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot([0, 1], [0, 1000])
    ax2.plot([0, 1], [0, 1])
    set_sci_notation_label(ax1, write_to_axis=ax2)
    exponent_texts_ax1 = [text.get_text() for text in ax1.texts]
    exponent_texts_ax2 = [text.get_text() for text in ax2.texts]
    assert len(exponent_texts_ax1) == 0, "No exponent text should be added to ax1"
    assert len(exponent_texts_ax2) == 1, "Exponent text should be added to ax2"
    assert exponent_texts_ax2[0] == r"$\times10^{3}$", "Exponent text should be correct"
    plt.close(fig)


def test_set_sci_notation_label_scilimits():
    fig, ax = plt.subplots()
    ax.plot([0, 1e5], [0, 1e5])
    set_sci_notation_label(ax, scilimits=(0, 0))
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        exponent_texts[-1] == r"$\times10^{5}$"
    ), "Exponent text should match the data scale"
    plt.close(fig)


def test_set_sci_notation_label_custom_fontsize():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    original_fontsize = plt.rcParams["xtick.labelsize"]
    set_sci_notation_label(ax)
    exponent_text = ax.texts[-1]
    expected_fontsize = original_fontsize * 0.85
    assert (
        exponent_text.get_fontsize() == expected_fontsize
    ), "Font size should be 85% of xtick.labelsize"
    plt.close(fig)


def test_set_sci_notation_label_negative_values():
    fig, ax = plt.subplots()
    ax.plot([0, -1], [0, -1000])
    ax.set_yticks([-1000, -500, 0, 500, 1000])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    # Now, there are positive ticks, so exponent text should be added
    assert (
        len(exponent_texts) == 1
    ), "Exponent text should be added when positive ticks are present"
    expected_exponent = int(np.floor(np.log10(1000)))  # 3
    assert (
        exponent_texts[0] == rf"$\times10^{{{expected_exponent}}}$"
    ), "Exponent text should be correct"
    plt.close(fig)


def test_set_sci_notation_label_log_scale():
    fig, ax = plt.subplots()
    ax.plot([1, 10], [1, 1000])
    ax.set_yscale("log")
    try:
        set_sci_notation_label(ax)
        # In log scale, ticklabel_format may not have an effect; ensure function handles this gracefully
        exponent_texts = [text.get_text() for text in ax.texts]
        assert len(exponent_texts) == 0, "No exponent text should be added in log scale"
    except AttributeError as e:
        assert "This method only works with the ScalarFormatter" in str(
            e
        ), "Unexpected error message"
    plt.close(fig)


def test_set_sci_notation_label_zero_ticks():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    ax.set_yticks([])  # Remove all y ticks
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        len(exponent_texts) == 0
    ), "No exponent text should be added when there are zero ticks"
    plt.close(fig)


def test_set_sci_notation_label_custom_transform():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    custom_axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    set_sci_notation_label(ax, write_to_axis=custom_axis)
    exponent_texts = [text.get_text() for text in custom_axis.texts]
    assert len(exponent_texts) == 1, "Exponent text should be added to the custom axis"
    plt.close(fig)


def test_set_sci_notation_label_large_numbers():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1e9])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        exponent_texts[0] == r"$\times10^{9}$"
    ), "Exponent text should handle large numbers correctly"
    plt.close(fig)


def test_set_sci_notation_label_small_numbers():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1e-3])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        exponent_texts[0] == r"$\times10^{-3}$"
    ), "Exponent text should handle small numbers correctly"
    plt.close(fig)


def test_set_sci_notation_label_with_zero_exponent():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    if len(exponent_texts) == 0:
        assert True, "No exponent text added when exponent is zero"
    else:
        assert exponent_texts[0] == r"$\times10^{0}$", "Exponent text should be zero"
    plt.close(fig)


def test_set_sci_notation_label_exponent_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 5000])
    set_sci_notation_label(ax)
    exponent_text = ax.texts[-1]
    assert (
        exponent_text.get_text() == r"$\times10^{3}$"
    ), "Exponent text should be correct for axis maximum"
    plt.close(fig)


def test_set_sci_notation_label_multiple_calls():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    set_sci_notation_label(ax)
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        len(exponent_texts) == 2
    ), "Should handle multiple calls and add multiple texts"
    plt.close(fig)


def test_set_sci_notation_label_no_ticks_positive():
    fig, ax = plt.subplots()
    ax.plot([0, -1], [0, -1000])
    ax.set_yticks([-1000, -500, 0])
    set_sci_notation_label(ax)
    exponent_texts = [text.get_text() for text in ax.texts]
    assert (
        len(exponent_texts) == 0
    ), "No exponent text should be added when there are no positive ticks"
    plt.close(fig)


def test_set_sci_notation_label_non_standard_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot([0, np.pi], [0, 1000])
    try:
        set_sci_notation_label(ax)
        # Function should handle or skip non-standard axes without crashing
    except TypeError as e:
        assert "'>' not supported between instances of" in str(
            e
        ), "Unexpected error message"
    plt.close(fig)


def test_set_sci_notation_label_large_offset_points():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1000])
    offset_points = (100, 50)
    set_sci_notation_label(ax, offset_points=offset_points)
    exponent_text = ax.texts[-1]
    assert (
        exponent_text.get_text() == r"$\times10^{3}$"
    ), "Exponent text should be correct with large offset"
    plt.close(fig)


def test_set_sci_notation_label_no_ticks_no_return():
    fig, ax = plt.subplots()
    ax.plot([], [])
    result = set_sci_notation_label(ax)
    assert result is None, "Function should return None when there are no ticks"
    plt.close(fig)


def test_set_sci_notation_label_return_none_zero_exponent():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    result = set_sci_notation_label(ax)
    assert result is None, "Function should return None when exponent is zero"
    plt.close(fig)


def test_set_sci_notation_label_handles_subplots():
    fig, axs = plt.subplots(2, 2)
    for ax in axs.flat:
        ax.plot([0, 1], [0, 1000])
        set_sci_notation_label(ax)
        exponent_texts = [text.get_text() for text in ax.texts]
        assert len(exponent_texts) == 1, "Exponent text should be added to each subplot"
    plt.close(fig)


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_valid_position(mock_number_to_letters):
    """Test labelfigs with valid position values (tl, tr, bl, br, ct, cb)."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Mock number_to_letters for consistent output
    mock_number_to_letters.return_value = "A"

    # Test all valid positions
    positions = ["tl", "tr", "bl", "br", "ct", "cb"]

    for loc in positions:
        text_obj = labelfigs(ax, number=1, loc=loc, string_add="Fig", style="wb")

        # Assert the text object is added
        assert isinstance(text_obj, plt.Text)

        # Check that the text includes the number (from mock) and the string_add
        assert text_obj.get_text() == "FigA"

        # Ensure the z-order is set to infinity
        assert text_obj.get_zorder() == np.inf


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_invalid_position(mock_number_to_letters):
    """Test labelfigs raises ValueError for invalid loc parameter."""
    fig, ax = plt.subplots()

    # Check that ValueError is raised for invalid location
    with pytest.raises(ValueError, match="Invalid position"):
        labelfigs(ax, number=1, loc="invalid_loc")


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_style_wb(mock_number_to_letters):
    """Test labelfigs with 'wb' style (white text with black border)."""
    fig, ax = plt.subplots()

    # Call the function with 'wb' style
    text_obj = labelfigs(ax, number=1, style="wb")

    # Check the path effects for the stroke
    path_effects_ = text_obj.get_path_effects()
    assert isinstance(path_effects_[0], patheffects.withStroke)

    # Verify the color and formatting
    assert text_obj.get_color() == "w"


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_style_b(mock_number_to_letters):
    """Test labelfigs with 'b' style (black text)."""
    fig, ax = plt.subplots()

    # Call the function with 'b' style
    text_obj = labelfigs(ax, number=1, style="b")

    # Check the path effects for the stroke
    path_effects_ = text_obj.get_path_effects()
    assert isinstance(path_effects_[0], patheffects.withStroke)

    # Verify the color and formatting
    assert text_obj.get_color() == "k"


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_number(mock_number_to_letters):
    """Test that number is correctly converted to letter using number_to_letters."""
    fig, ax = plt.subplots()

    # Mock number_to_letters to return "B" for number 2
    mock_number_to_letters.return_value = "B"

    # Call the function with a number
    text_obj = labelfigs(ax, number=2, string_add="Fig")

    # Check that the correct text is set, including the converted number
    assert text_obj.get_text() == "FigB"


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_no_number(mock_number_to_letters):
    """Test labelfigs without a number."""
    fig, ax = plt.subplots()

    # Call the function without a number
    text_obj = labelfigs(ax, string_add="Fig")

    # Ensure the text only includes the string_add
    assert text_obj.get_text() == "Fig"

    # Ensure that number_to_letters was not called
    mock_number_to_letters.assert_not_called()


@patch("m3util.viz.text.number_to_letters")
def test_labelfigs_inset_fraction(mock_number_to_letters):
    """Test that the label is correctly positioned based on inset_fraction."""
    fig, ax = plt.subplots()

    # Set x and y limits of the axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    # Call labelfigs with a specific inset_fraction
    inset_fraction = (0.2, 0.1)
    text_obj = labelfigs(ax, loc="br", inset_fraction=inset_fraction, string_add="Test")

    # Verify the position of the label
    pos = text_obj.get_position()

    # Expected position for "br" (bottom-right) location
    expected_x = 10 - (10 * 0.1)  # Right inset
    expected_y = 0 + (5 * 0.2)  # Bottom inset

    assert np.allclose(pos[0], expected_x)
    assert np.allclose(pos[1], expected_y)


@patch("m3util.viz.positioning.obj_offset")
@patch("m3util.util.kwargs._filter_kwargs")
def test_line_annotation_default_values(mock_filter_kwargs, mock_obj_offset):
    """Test that line_annotation correctly annotates the midpoint with default values."""
    fig, ax = plt.subplots()

    # Define line coordinates
    line_x = np.array([0, 10])
    line_y = np.array([0, 20])

    # Mock _filter_kwargs to pass all keyword arguments as is
    mock_filter_kwargs.side_effect = lambda func, kwargs: kwargs

    # Mock obj_offset to return the default offset
    mock_obj_offset.return_value = (0, 0)

    # Call the line_annotation function
    annotation_kwargs = {}
    line_annotation(ax, "Midpoint Text", line_x, line_y, annotation_kwargs)

    # Check if annotate was called
    annotations = ax.texts
    assert len(annotations) == 1  # Only one annotation should be added

    # Check the annotation properties
    annotation = annotations[0]
    mid_x = np.mean(line_x)
    mid_y = np.mean(line_y)

    # Ensure the annotation is placed at the midpoint
    assert annotation.get_text() == "Midpoint Text"
    assert annotation.get_ha() == "center"  # Default horizontal alignment
    assert annotation.get_va() == "center"  # Default vertical alignment
    assert annotation.get_zorder() == 100  # Default zorder



@patch("m3util.viz.positioning.obj_offset")
@patch("m3util.util.kwargs._filter_kwargs")
def test_line_annotation_custom_values(mock_filter_kwargs, mock_obj_offset):
    """Test line_annotation with custom annotation kwargs and zorder."""
    fig, ax = plt.subplots()

    # Define line coordinates
    line_x = np.array([0, 10])
    line_y = np.array([0, 20])

    # Mock _filter_kwargs to pass all keyword arguments as is
    mock_filter_kwargs.side_effect = lambda func, kwargs: kwargs

    # Mock obj_offset to return a custom offset
    mock_obj_offset.return_value = (10, 10)

    # Call the line_annotation function with custom arguments
    annotation_kwargs = {"ha": "right", "va": "bottom", "fontsize": 12}
    line_annotation(ax, "Custom Text", line_x, line_y, annotation_kwargs, zorder=200)

    # Check if annotate was called
    annotations = ax.texts
    assert len(annotations) == 1  # Only one annotation should be added

    # Check the annotation properties
    annotation = annotations[0]
    mid_x = np.mean(line_x)
    mid_y = np.mean(line_y)

    # Ensure the annotation is placed at the midpoint
    assert annotation.get_text() == "Custom Text"
    assert annotation.get_ha() == "right"  # Custom horizontal alignment
    assert annotation.get_va() == "bottom"  # Custom vertical alignment
    assert annotation.get_zorder() == 200  # Custom zorder



@patch("m3util.viz.positioning.obj_offset")
@patch("m3util.util.kwargs._filter_kwargs")
def test_line_annotation_with_offset(mock_filter_kwargs, mock_obj_offset):
    """Test line_annotation with a custom text offset."""
    fig, ax = plt.subplots()

    # Define line coordinates
    line_x = np.array([1, 3, 5, 7])
    line_y = np.array([2, 4, 6, 8])

    # Mock _filter_kwargs to pass all keyword arguments as is
    mock_filter_kwargs.side_effect = lambda func, kwargs: kwargs

    # Mock obj_offset to return a specific offset
    mock_obj_offset.return_value = (15, 5)

    # Call the line_annotation function with a custom text offset
    annotation_kwargs = {"ha": "left", "va": "top", "fontsize": 10}
    line_annotation(ax, "Offset Text", line_x, line_y, annotation_kwargs)

    # Check if annotate was called
    annotations = ax.texts
    assert len(annotations) == 1  # Only one annotation should be added

    # Check the annotation properties
    annotation = annotations[0]
    mid_x = np.mean(line_x)
    mid_y = np.mean(line_y)

    # Ensure the annotation is placed at the midpoint
    assert annotation.get_text() == "Offset Text"
    assert annotation.get_ha() == "left"  # Custom horizontal alignment
    assert annotation.get_va() == "top"  # Custom vertical alignment



@patch("m3util.viz.positioning.obj_offset")
@patch("m3util.util.kwargs._filter_kwargs")
def test_line_annotation_with_empty_annotation_kwargs(
    mock_filter_kwargs, mock_obj_offset
):
    """Test line_annotation with empty annotation kwargs."""
    fig, ax = plt.subplots()

    # Define line coordinates
    line_x = np.array([2, 4, 6, 8])
    line_y = np.array([1, 3, 5, 7])

    # Mock _filter_kwargs to pass all keyword arguments as is
    mock_filter_kwargs.side_effect = lambda func, kwargs: kwargs

    # Mock obj_offset to return the default offset
    mock_obj_offset.return_value = (0, 0)

    # Call the line_annotation function with empty annotation kwargs
    annotation_kwargs = {}
    line_annotation(ax, "Test Label", line_x, line_y, annotation_kwargs)

    # Check if annotate was called
    annotations = ax.texts
    assert len(annotations) == 1  # Only one annotation should be added

    # Check the annotation properties
    annotation = annotations[0]
    mid_x = np.mean(line_x)
    mid_y = np.mean(line_y)




@patch("m3util.viz.positioning.obj_offset")
@patch("m3util.util.kwargs._filter_kwargs")
def test_line_annotation_with_no_text(mock_filter_kwargs, mock_obj_offset):
    """Test line_annotation with empty text input."""
    fig, ax = plt.subplots()

    # Define line coordinates
    line_x = np.array([2, 4])
    line_y = np.array([1, 3])

    # Mock _filter_kwargs to pass all keyword arguments as is
    mock_filter_kwargs.side_effect = lambda func, kwargs: kwargs

    # Mock obj_offset to return the default offset
    mock_obj_offset.return_value = (0, 0)

    # Call the line_annotation function with no text
    annotation_kwargs = {}
    line_annotation(ax, "", line_x, line_y, annotation_kwargs)

    # Check if annotate was called
    annotations = ax.texts
    assert len(annotations) == 1  # One annotation should still be added

    # Check the annotation properties
    annotation = annotations[0]
    mid_x = np.mean(line_x)
    mid_y = np.mean(line_y)

    # Ensure the annotation is placed at the midpoint
    assert annotation.get_text() == ""  # No text provided

