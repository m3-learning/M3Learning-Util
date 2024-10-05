import pytest
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Rectangle, ConnectionPatch
from unittest.mock import patch, MagicMock
from m3util.viz.layout import (
    plot_into_graph,
    subfigures,
    add_box,
    inset_connector,
    path_maker,
    layout_fig,
    embedding_maps,
    imagemap,
    find_nearest,
    combine_lines,
    scalebar,
    Axis_Ratio,
    get_axis_range,
    set_axis,
    add_scalebar,
    get_axis_pos_inches,
    get_closest_point,
    span_to_axis,
    draw_line_with_text,
    FigDimConverter,
)
from m3util.viz.text import add_text_to_figure, labelfigs, number_to_letters

@pytest.fixture
def sample_figure():
    fig, ax = plt.subplots()
    return fig, ax


def test_plot_into_graph(sample_figure):
    fig, ax = sample_figure
    fig_test, ax_test = plt.subplots()
    plot_into_graph(ax, fig_test)
    assert len(ax.get_images()) > 0, "The image should be plotted into the axes."


def test_subfigures():
    fig, ax = subfigures(2, 2)
    assert len(ax) == 4, "Should create 4 subfigures."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."


def test_add_text_to_figure(sample_figure):
    fig, ax = sample_figure
    add_text_to_figure(fig, "Test Text", (1, 1))
    assert len(fig.texts) == 1, "Text should be added to the figure."


def test_add_box(sample_figure):
    fig, ax = sample_figure
    add_box(ax, (0.1, 0.1, 0.4, 0.4))
    assert any(
        isinstance(patch, Rectangle) for patch in ax.patches
    ), "A rectangle should be added to the axes."


def test_inset_connector(sample_figure):
    fig, ax = sample_figure
    fig2, ax2 = plt.subplots()
    inset_connector(fig, ax, ax2)
    assert (
        len(fig.findobj(ConnectionPatch)) > 0
    ), "A connection patch should be added between axes."


def test_path_maker(sample_figure):
    fig, ax = sample_figure
    path_maker(ax, (0.1, 0.4, 0.1, 0.4), "blue", "red", "-", 2)
    assert len(ax.patches) == 1, "A path patch should be added to the axes."


def test_layout_fig():
    fig, axes = layout_fig(4)
    assert len(axes) == 4, "Should create 4 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."


def test_embedding_maps():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    embedding_maps(data, image)
    plt.close()


def test_imagemap(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(100).reshape(10, 10)
    imagemap(ax, data)
    assert len(ax.get_images()) > 0, "The data should be plotted as an image map."


def test_find_nearest():
    array = np.array([1, 3, 7, 8, 9])
    idx = find_nearest(array, 5, 2)
    assert len(idx) == 2, "Should find the two nearest neighbors."


def test_combine_lines(sample_figure):
    fig, ax = sample_figure
    ax.plot([1, 2], [3, 4], label="Line 1")
    ax2 = fig.add_subplot(111)
    ax2.plot([2, 3], [4, 5], label="Line 2")
    lines, labels = combine_lines(ax, ax2)
    assert len(lines) == 2, "Should combine the lines from both axes."
    assert len(labels) == 2, "Should combine the labels from both axes."


def test_labelfigs(sample_figure):
    fig, ax = sample_figure
    labelfigs(ax, number=1)
    assert len(ax.texts) > 0, "Label should be added to the figure."


def test_number_to_letters():
    assert (
        number_to_letters(0) == "a"
    ), "Number to letter conversion should return 'a' for 0."
    assert (
        number_to_letters(25) == "z"
    ), "Number to letter conversion should return 'z' for 25."
    assert (
        number_to_letters(26) == "aa"
    ), "Number to letter conversion should return 'aa' for 26."


def test_scalebar(sample_figure):
    fig, ax = sample_figure
    scalebar(ax, 100, 10)
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."


def test_Axis_Ratio(sample_figure):
    fig, ax = sample_figure
    Axis_Ratio(ax, ratio=2)
    assert ax.get_aspect() == 2, "The aspect ratio should be set to 2."


def test_get_axis_range(sample_figure):
    fig, ax = sample_figure
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    range_ = get_axis_range([ax])
    assert range_ == [0, 10, 0, 20], "Should return the correct axis range."


def test_set_axis(sample_figure):
    fig, ax = sample_figure
    set_axis([ax], [0, 10, 0, 20])
    assert ax.get_xlim() == (0, 10), "The x-axis range should be set correctly."
    assert ax.get_ylim() == (0, 20), "The y-axis range should be set correctly."


def test_add_scalebar(sample_figure):
    fig, ax = sample_figure
    add_scalebar(ax, {"width": 100, "scale length": 10, "units": "nm"})
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."


def test_get_axis_pos_inches(sample_figure):
    fig, ax = sample_figure
    pos_inches = get_axis_pos_inches(fig, ax)
    assert isinstance(
        pos_inches, np.ndarray
    ), "Position should be returned as a numpy array."


def test_FigDimConverter():
    converter = FigDimConverter((10, 5))
    inches = converter.to_inches((0.5, 0.5, 0.5, 0.5))
    assert inches == (
        5,
        2.5,
        5,
        2.5,
    ), "Should convert relative dimensions to inches correctly."
    relative = converter.to_relative((5, 2.5, 5, 2.5))
    assert relative == (
        0.5,
        0.5,
        0.5,
        0.5,
    ), "Should convert inches to relative dimensions correctly."


def test_add_box():
    fig, ax = plt.subplots()

    # Define the position of the box
    pos = (0.1, 0.2, 0.4, 0.6)

    # Call the add_box function
    add_box(ax, pos, edgecolor="red", facecolor="none", linewidth=2)

    # Retrieve the added patches from the axes
    patches = ax.patches

    # Check that exactly one rectangle was added
    assert len(patches) == 1, "Expected exactly one rectangle to be added to the axes"

    # Check that the added patch is a Rectangle
    assert isinstance(patches[0], Rectangle), "The added patch should be a Rectangle"

    # Check that the position and size of the rectangle are correct
    rect = patches[0]
    assert rect.get_x() == pos[0], "The x-position of the rectangle is incorrect"
    assert rect.get_y() == pos[1], "The y-position of the rectangle is incorrect"
    assert (
        rect.get_width() == pos[2] - pos[0]
    ), "The width of the rectangle is incorrect"
    assert (
        rect.get_height() == pos[3] - pos[1]
    ), "The height of the rectangle is incorrect"

    # Check that the rectangle has the correct properties
    assert rect.get_edgecolor() == (
        1.0,
        0.0,
        0.0,
        1.0,
    ), "The edge color of the rectangle is incorrect"
    assert rect.get_facecolor() == (
        0.0,
        0.0,
        0.0,
        0.0,
    ), "The face color of the rectangle is incorrect"
    assert rect.get_linewidth() == 2, "The line width of the rectangle is incorrect"


def test_subfigures_default():
    # Test with default size and gaps
    nrows, ncols = 2, 3
    fig, ax = subfigures(nrows, ncols)

    # Check if the number of axes is correct
    assert (
        len(ax) == nrows * ncols
    ), f"Expected {nrows * ncols} subfigures, but got {len(ax)}."

    # Check the size of the figure
    expected_figsize = (1.25 * ncols + 0.8 * ncols, 1.25 * nrows + 0.33 * nrows)
    assert fig.get_size_inches() == pytest.approx(
        expected_figsize
    ), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."


def test_subfigures_custom_size_and_gaps():
    # Test with custom size and gaps
    nrows, ncols = 3, 2
    custom_size = (2.0, 2.0)
    custom_gaps = (1.0, 0.5)
    fig, ax = subfigures(nrows, ncols, size=custom_size, gaps=custom_gaps)

    # Check if the number of axes is correct
    assert (
        len(ax) == nrows * ncols
    ), f"Expected {nrows * ncols} subfigures, but got {len(ax)}."

    # Check the size of the figure
    expected_figsize = (
        custom_size[0] * ncols + custom_gaps[0] * ncols,
        custom_size[1] * nrows + custom_gaps[1] * nrows,
    )
    assert fig.get_size_inches() == pytest.approx(
        expected_figsize
    ), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."


def test_subfigures_custom_figsize():
    # Test with a custom figsize
    nrows, ncols = 2, 2
    custom_figsize = (8, 6)
    fig, ax = subfigures(nrows, ncols, figsize=custom_figsize)

    # Check if the number of axes is correct
    assert (
        len(ax) == nrows * ncols
    ), f"Expected {nrows * ncols} subfigures, but got {len(ax)}."

    # Check the size of the figure
    assert np.allclose(
        fig.get_size_inches(), custom_figsize
    ), f"Expected figure size {custom_figsize}, but got {fig.get_size_inches()}."


def test_subfigures_size():
    # Test with custom sizes
    nrows, ncols = 2, 3
    custom_size = (2.0, 2.0)
    fig, ax = subfigures(nrows, ncols, size=custom_size)

    # Check if the number of axes is correct
    assert (
        len(ax) == nrows * ncols
    ), f"Expected {nrows * ncols} subfigures, but got {len(ax)}."

    # Check the size of the figure
    expected_figsize = (
        custom_size[0] * ncols + 0.8 * ncols,
        custom_size[1] * nrows + 0.33 * nrows,
    )
    assert np.allclose(
        fig.get_size_inches(), expected_figsize
    ), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."


def test_subfigures_gaps():
    # Test with custom gaps
    nrows, ncols = 3, 2
    custom_gaps = (1.0, 0.5)
    fig, ax = subfigures(nrows, ncols, gaps=custom_gaps)

    # Check if the number of axes is correct
    assert (
        len(ax) == nrows * ncols
    ), f"Expected {nrows * ncols} subfigures, but got {len(ax)}."

    # Check the size of the figure
    expected_figsize = (
        1.25 * ncols + custom_gaps[0] * ncols,
        1.25 * nrows + custom_gaps[1] * nrows,
    )
    assert np.allclose(
        fig.get_size_inches(), expected_figsize
    ), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."


def test_subfigures_size_and_gaps():
    # Test with custom sizes and gaps
    nrows, ncols = 2, 2
    custom_size = (2.5, 2.0)
    custom_gaps = (0.5, 1.0)
    fig, ax = subfigures(nrows, ncols, size=custom_size, gaps=custom_gaps)

    # Check if the number of axes is correct
    assert (
        len(ax) == nrows * ncols
    ), f"Expected {nrows * ncols} subfigures, but got {len(ax)}."

    # Check the size of the figure
    expected_figsize = (
        custom_size[0] * ncols + custom_gaps[0] * ncols,
        custom_size[1] * nrows + custom_gaps[1] * nrows,
    )
    assert np.allclose(
        fig.get_size_inches(), expected_figsize
    ), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."


def test_get_closest_point_x_axis():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 20, 30, 40, 50])
    value = 3.7
    closest_x, closest_y = get_closest_point(x_data, y_data, value, axis="x")
    assert closest_x == 4, "The closest x value should be 4."
    assert closest_y == 40, "The closest y value should be 40."


def test_get_closest_point_y_axis():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 20, 30, 40, 50])
    value = 25
    closest_x, closest_y = get_closest_point(x_data, y_data, value, axis="y")
    assert closest_x == 2, "The closest x value should be 2."
    assert closest_y == 20, "The closest y value should be 20."


def test_get_closest_point_invalid_axis():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 20, 30, 40, 50])
    value = 3
    try:
        get_closest_point(x_data, y_data, value, axis="z")
    except ValueError as e:
        assert (
            str(e) == "axis must be 'x' or 'y'"
        ), "Should raise ValueError for invalid axis."


def test_get_closest_point_different_lengths():
    x_data = np.array([1, 2, 3])
    y_data = np.array([10, 20])
    value = 2
    try:
        get_closest_point(x_data, y_data, value, axis="x")
    except ValueError as e:
        assert (
            str(e) == "x_data and y_data must have the same shape."
        ), "Should raise ValueError for different lengths."


def test_get_closest_point_exact_match():
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 20, 30, 40, 50])
    value = 3
    closest_x, closest_y = get_closest_point(x_data, y_data, value, axis="x")
    assert closest_x == 3, "The closest x value should be 3."
    assert closest_y == 30, "The closest y value should be 30."


def test_inset_connector_default_coords():
    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    inset_connector(fig, ax1, ax2)
    assert (
        len(fig.findobj(ConnectionPatch)) > 0
    ), "A connection patch should be added between axes with default coordinates."


def test_inset_connector_with_kwargs():
    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    inset_connector(fig, ax1, ax2, color="red", linestyle="--")
    connection_patches = fig.findobj(ConnectionPatch)
    assert (
        len(connection_patches) > 0
    ), "A connection patch should be added between axes with additional kwargs."
    assert connection_patches[0].get_edgecolor() == (
        1.0,
        0.0,
        0.0,
        1.0,
    ), "The connection patch should have the correct color."
    assert (
        connection_patches[0].get_linestyle() == "--"
    ), "The connection patch should have the correct linestyle."


def test_layout_fig_default():
    fig, axes = layout_fig(6)
    assert len(axes) == 6, "Should create 6 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert all(isinstance(ax, plt.Axes) for ax in axes), "All elements should be Axes."


def test_layout_fig_custom_mod():
    fig, axes = layout_fig(6, mod=2)
    assert len(axes) == 6, "Should create 6 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert all(isinstance(ax, plt.Axes) for ax in axes), "All elements should be Axes."
    assert fig.get_size_inches() == pytest.approx(
        (6, 9)
    ), "Figure size should be (6, 9)."


def test_layout_fig_custom_figsize():
    fig, axes = layout_fig(6, figsize=(12, 8))
    assert len(axes) == 6, "Should create 6 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert all(isinstance(ax, plt.Axes) for ax in axes), "All elements should be Axes."
    assert fig.get_size_inches() == pytest.approx(
        (12, 8)
    ), "Figure size should be (12, 8)."


def test_layout_fig_custom_layout():
    fig, axes = layout_fig(6, layout="tight")
    assert len(axes) == 6, "Should create 6 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert all(isinstance(ax, plt.Axes) for ax in axes), "All elements should be Axes."


def test_layout_fig_extra_axes_deleted():
    fig, axes = layout_fig(5, mod=3)
    assert len(axes) == 5, "Should create 5 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert all(isinstance(ax, plt.Axes) for ax in axes), "All elements should be Axes."
    assert len(fig.get_axes()) == 5, "Should only have 5 axes in the figure."


def test_embedding_maps_default():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    embedding_maps(data, image)
    plt.close()


def test_embedding_maps_with_colorbar():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    embedding_maps(data, image, colorbar_shown=True)
    plt.close()


def test_embedding_maps_without_colorbar():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    embedding_maps(data, image, colorbar_shown=False)
    plt.close()


def test_embedding_maps_with_clim():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    c_lim = [0, 1]
    embedding_maps(data, image, c_lim=c_lim)
    plt.close()


def test_embedding_maps_with_title():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    title = "Test Title"
    embedding_maps(data, image, title=title)
    plt.close()


def test_embedding_maps_with_mod():
    data = np.random.rand(100, 4)
    image = np.random.rand(10, 10)
    mod = 2
    embedding_maps(data, image, mod=mod)
    plt.close()


def test_imagemap_default(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(10, 10)
    imagemap(ax, data)
    assert len(ax.get_images()) > 0, "The data should be plotted as an image map."
    assert (
        ax.get_images()[0].get_cmap().name == "viridis"
    ), "Default colormap should be 'viridis'."


def test_imagemap_with_clim(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(10, 10)
    clim = (0.2, 0.8)
    imagemap(ax, data, clim=clim)
    im = ax.get_images()[0]
    assert im.get_clim() == clim, "Color limits should be set correctly."


def test_imagemap_with_colorbar(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(10, 10)
    imagemap(ax, data, colorbars=True)
    assert len(fig.axes) > 1, "Colorbar should be added to the figure."


def test_imagemap_without_colorbar(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(10, 10)
    imagemap(ax, data, colorbars=False)
    assert len(fig.axes) == 1, "No colorbar should be added to the figure."


def test_imagemap_custom_cmap(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(10, 10)
    custom_cmap = "plasma"
    imagemap(ax, data, cmap_=custom_cmap)
    assert (
        ax.get_images()[0].get_cmap().name == custom_cmap
    ), f"Colormap should be '{custom_cmap}'."


def test_imagemap_1d_data(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(100)
    imagemap(ax, data)
    assert (
        len(ax.get_images()) > 0
    ), "1D data should be reshaped and plotted as an image map."
    assert ax.get_images()[0].get_array().shape == (
        10,
        10,
    ), "1D data should be reshaped to 2D."


def test_imagemap_with_divider(sample_figure):
    fig, ax = sample_figure
    data = np.random.rand(10, 10)
    imagemap(ax, data, divider_=True)
    assert len(fig.axes) > 1, "Divider should be added to the figure."


def test_find_nearest_single_neighbor():
    array = np.array([1, 3, 7, 8, 9])
    idx = find_nearest(array, 5, 1)
    assert len(idx) == 1, "Should find the single nearest neighbor."
    assert array[idx[0]] == 3, "The nearest neighbor should be 3."


def test_find_nearest_multiple_neighbors():
    array = np.array([1, 3, 7, 8, 9])
    idx = find_nearest(array, 5, 2)
    assert len(idx) == 2, "Should find the two nearest neighbors."
    assert set(array[idx]) == {3, 7}, "The nearest neighbors should be 3 and 7."


def test_find_nearest_exact_match():
    array = np.array([1, 3, 5, 7, 9])
    idx = find_nearest(array, 5, 1)
    assert len(idx) == 1, "Should find the single nearest neighbor."
    assert array[idx[0]] == 5, "The nearest neighbor should be 5."


def test_find_nearest_large_averaging_number():
    array = np.array([1, 3, 7, 8, 9])
    idx = find_nearest(array, 5, 10)
    assert len(idx) == len(
        array
    ), "Should return all elements when averaging number is larger than array length."


def test_find_nearest_empty_array():
    array = np.array([])
    idx = find_nearest(array, 5, 1)
    assert len(idx) == 0, "Should return an empty array when input array is empty."


def test_combine_lines_single_axis(sample_figure):
    fig, ax = sample_figure
    ax.plot([1, 2], [3, 4], label="Line 1")
    lines, labels = combine_lines(ax)
    assert len(lines) == 1, "Should combine the lines from the single axis."
    assert len(labels) == 1, "Should combine the labels from the single axis."
    assert labels[0] == "Line 1", "The label should be 'Line 1'."


def test_combine_lines_multiple_axes(sample_figure):
    fig, ax = sample_figure
    ax.plot([1, 2], [3, 4], label="Line 1")
    ax2 = fig.add_subplot(111)
    ax2.plot([2, 3], [4, 5], label="Line 2")
    lines, labels = combine_lines(ax, ax2)
    assert len(lines) == 2, "Should combine the lines from both axes."
    assert len(labels) == 2, "Should combine the labels from both axes."
    assert labels == ["Line 1", "Line 2"], "The labels should be 'Line 1' and 'Line 2'."


def test_combine_lines_empty_axes(sample_figure):
    fig, ax = sample_figure
    ax2 = fig.add_subplot(111)
    lines, labels = combine_lines(ax, ax2)
    assert len(lines) == 0, "Should handle empty axes without errors."
    assert len(labels) == 0, "Should handle empty axes without errors."


def test_scalebar_bottom_right():
    fig, ax = plt.subplots()
    image_size = 100
    scale_size = 10
    scalebar(ax, image_size, scale_size, units="nm", loc="br")
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."
    assert any(
        isinstance(patch, patches.PathPatch) for patch in ax.patches
    ), "A PathPatch should be added to the axes."
    assert any(
        isinstance(text, plt.Text) and text.get_text() == "10 nm" for text in ax.texts
    ), "The scalebar label should be added to the axes."


def test_scalebar_top_right():
    fig, ax = plt.subplots()
    image_size = 100
    scale_size = 10
    scalebar(ax, image_size, scale_size, units="nm", loc="tr")
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."
    assert any(
        isinstance(patch, patches.PathPatch) for patch in ax.patches
    ), "A PathPatch should be added to the axes."
    assert any(
        isinstance(text, plt.Text) and text.get_text() == "10 nm" for text in ax.texts
    ), "The scalebar label should be added to the axes."


def test_scalebar_custom_units():
    fig, ax = plt.subplots()
    image_size = 100
    scale_size = 10
    scalebar(ax, image_size, scale_size, units="µm", loc="br")
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."
    assert any(
        isinstance(patch, patches.PathPatch) for patch in ax.patches
    ), "A PathPatch should be added to the axes."
    assert any(
        isinstance(text, plt.Text) and text.get_text() == "10 µm" for text in ax.texts
    ), "The scalebar label should be added to the axes."


def test_scalebar_different_image_size():
    fig, ax = plt.subplots()
    image_size = 200
    scale_size = 20
    scalebar(ax, image_size, scale_size, units="nm", loc="br")
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."
    assert any(
        isinstance(patch, patches.PathPatch) for patch in ax.patches
    ), "A PathPatch should be added to the axes."
    assert any(
        isinstance(text, plt.Text) and text.get_text() == "20 nm" for text in ax.texts
    ), "The scalebar label should be added to the axes."


###### Span to axis #####

import pytest
from unittest.mock import MagicMock


# Mock the get_closest_point function for testing purposes
def mock_get_closest_point(x_data, y_data, value, axis="x"):
    # For testing, return a simple value depending on the axis
    if axis == "x":
        return value, 10  # mock a y-value of 10 for any x
    elif axis == "y":
        return 10, value  # mock an x-value of 10 for any y


# Test cases
@pytest.fixture
def mock_axis():
    # Create a mock axis object with predefined x and y limits
    ax = MagicMock()
    ax.get_xlim.return_value = (0, 100)  # x-axis limits
    ax.get_ylim.return_value = (0, 50)  # y-axis limits
    return ax


# Patch the get_closest_point function
@pytest.mark.parametrize(
    "connect_to, expected_x, expected_y",
    [
        ("left", [50, 0], [10, 10]),  # Connecting to the left y-axis
        ("right", [50, 100], [10, 10]),  # Connecting to the right y-axis
        ("bottom", [10, 10], [20, 0]),  # Connecting to the bottom x-axis
        ("top", [10, 10], [20, 50]),  # Connecting to the top x-axis
    ],
)
@patch("m3util.viz.layout.get_closest_point", side_effect=mock_get_closest_point)
def test_span_to_axis(
    mock_get_closest_point, mock_axis, connect_to, expected_x, expected_y
):
    x_data = [10, 50, 90]
    y_data = [5, 10, 15]
    value = 50 if connect_to in ["left", "right"] else 20

    # Call the span_to_axis function
    line_x, line_y = span_to_axis(mock_axis, value, x_data, y_data, connect_to)

    # Assert that the output matches the expected values
    assert line_x == expected_x
    assert line_y == expected_y


@patch("m3util.viz.layout.get_closest_point", side_effect=mock_get_closest_point)
def test_span_to_axis_invalid_connect_to(mock_get_closest_point, mock_axis):
    x_data = [10, 50, 90]
    y_data = [5, 10, 15]
    value = 50

    # Test invalid `connect_to` value
    with pytest.raises(ValueError, match="Invalid connect_to value"):
        span_to_axis(mock_axis, value, x_data, y_data, connect_to="invalid")


@patch("m3util.viz.layout.get_closest_point", side_effect=mock_get_closest_point)
def test_span_to_axis_invalid_connect_to_for_axis(mock_get_closest_point, mock_axis):
    x_data = [10, 50, 90]
    y_data = [5, 10, 15]
    value = 50

    # Test invalid connect_to value for axis-specific
    with pytest.raises(
        ValueError, match="Invalid connect_to value. Choose 'left', 'right'"
    ):
        span_to_axis(mock_axis, value, x_data, y_data, connect_to="center")


# Mock line_annotation since it's used internally
def mock_line_annotation(ax, text, line_x, line_y, annotation_kwargs, zorder=2):
    # A simple mock to bypass actual text annotation
    pass


@pytest.fixture
def mock_axis():
    # Create a mock axis object with predefined x and y limits
    ax = MagicMock()
    ax.get_xlim.return_value = (0, 100)  # x-axis limits
    ax.get_ylim.return_value = (0, 50)  # y-axis limits
    return ax


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_vertical_full(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 50

    # Call the function for a vertical line spanning the full y-axis
    draw_line_with_text(mock_axis, x_data, y_data, value, axis="x", span="full")

    # Assert that ax.plot is called with the correct line coordinates
    mock_axis.plot.assert_called_once_with([value, value], [0, 50], zorder=2)


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_horizontal_full(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 10

    # Call the function for a horizontal line spanning the full x-axis
    draw_line_with_text(mock_axis, x_data, y_data, value, axis="y", span="full")

    # Assert that ax.plot is called with the correct line coordinates
    mock_axis.plot.assert_called_once_with([0, 100], [value, value], zorder=2)


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_span_data_vertical(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 50

    # Call the function for a vertical line between closest y-values
    draw_line_with_text(mock_axis, x_data, y_data, value, axis="x", span="data")

    # Assert that ax.plot is called with the correct line coordinates between y1 and y2
    mock_axis.plot.assert_called_once_with([value, value], [10, 10], zorder=2)


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_span_data_horizontal(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 10

    # Call the function for a horizontal line between closest x-values
    draw_line_with_text(mock_axis, x_data, y_data, value, axis="y", span="data")

    # Assert that ax.plot is called with the correct line coordinates between x1 and x2
    mock_axis.plot.assert_called_once_with([10, 50], [value, value], zorder=2)


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_invalid_axis(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 10

    # Test invalid axis value
    with pytest.raises(ValueError, match="axis must be 'x' or 'y'"):
        draw_line_with_text(mock_axis, x_data, y_data, value, axis="z", span="full")


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_invalid_span(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 10

    # Test invalid span value
    with pytest.raises(ValueError, match="span must be 'full' or 'data'"):
        draw_line_with_text(mock_axis, x_data, y_data, value, axis="x", span="invalid")


@patch("m3util.viz.layout.line_annotation", side_effect=mock_line_annotation)
def test_draw_line_with_text_data_outside_range(mock_line_annotation, mock_axis):
    x_data = np.array([10, 50, 90])
    y_data = np.array([5, 10, 15])
    value = 100  # Value outside the range of x_data

    # Test value outside the range of data points
    with pytest.raises(ValueError, match="Value is outside the range of x_data."):
        draw_line_with_text(mock_axis, x_data, y_data, value, axis="x", span="data")
