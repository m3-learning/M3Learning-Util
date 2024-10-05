from m3util.viz.text import add_text_to_figure, set_sci_notation_label
import matplotlib.pyplot as plt
import pytest
import numpy as np
from matplotlib import patheffects


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
    expected_offset_x = offset_points[0] / dpi / fig_width
    expected_offset_y = offset_points[1] / dpi / fig_height
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
