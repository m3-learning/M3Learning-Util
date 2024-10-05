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
    layout_subfigures_inches,
    get_zorders,
    FigDimConverter,
)
from m3util.viz.text import add_text_to_figure, labelfigs, number_to_letters
from matplotlib import patheffects


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


def test_add_box_():
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
def mock_axis_v2():
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


@patch("m3util.viz.layout.ConnectionPatch", autospec=True)
def test_inset_connector_default_coords_(mock_connection_patch):
    """Test with default coordinates (coord1 and coord2 are None)."""
    # Create a figure and two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Set limits for ax1 and ax2
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 20)
    ax2.set_xlim(5, 15)
    ax2.set_ylim(10, 30)

    # Call the function with default coord1 and coord2 (None)
    inset_connector(fig, ax1, ax2)

    # Check if ConnectionPatch is called twice for both points
    assert mock_connection_patch.call_count == 2

    # Verify the first call's arguments for p1 and p2
    call_args = mock_connection_patch.call_args_list[0]
    p1 = call_args[1]["xyA"]
    p2 = call_args[1]["xyB"]

    # p1 should be the bottom-left corner of ax1 and p2 the bottom-left of ax2
    assert p1 == (0, 0)  # xlim[0], ylim[0] for ax1
    assert p2 == (5, 10)  # xlim[0], ylim[0] for ax2

    # Verify the second call's arguments for p1 and p2
    call_args = mock_connection_patch.call_args_list[1]
    p1 = call_args[1]["xyA"]
    p2 = call_args[1]["xyB"]

    # p1 should be the top-left corner of ax1 and p2 the top-left of ax2
    assert p1 == (0, 20)  # xlim[0], ylim[1] for ax1
    assert p2 == (5, 30)  # xlim[0], ylim[1] for ax2


@patch("m3util.viz.layout.ConnectionPatch", autospec=True)
def test_inset_connector_custom_coords(mock_connection_patch):
    """Test with custom coord1 and coord2."""
    # Create a figure and two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Custom coordinates for the connection points
    coord1 = [(1, 2), (3, 4)]
    coord2 = [(6, 7), (8, 9)]

    # Call the function with custom coordinates
    inset_connector(fig, ax1, ax2, coord1=coord1, coord2=coord2)

    # Check if ConnectionPatch is called twice for both custom points
    assert mock_connection_patch.call_count == 2

    # Verify the first call's arguments for p1 and p2
    call_args = mock_connection_patch.call_args_list[0]
    p1 = call_args[1]["xyA"]
    p2 = call_args[1]["xyB"]

    assert p1 == (1, 2)  # Custom coord1[0]
    assert p2 == (6, 7)  # Custom coord2[0]

    # Verify the second call's arguments for p1 and p2
    call_args = mock_connection_patch.call_args_list[1]
    p1 = call_args[1]["xyA"]
    p2 = call_args[1]["xyB"]

    assert p1 == (3, 4)  # Custom coord1[1]
    assert p2 == (8, 9)  # Custom coord2[1]


@patch("m3util.viz.layout.ConnectionPatch", autospec=True)
def test_inset_connector_with_kwargs_(mock_connection_patch):
    """Test if additional keyword arguments are passed to ConnectionPatch."""
    # Create a figure and two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Custom coordinates
    coord1 = [(1, 2)]
    coord2 = [(6, 7)]

    # Additional keyword arguments for the ConnectionPatch
    kwargs = {"color": "red", "linestyle": "--"}

    # Call the function with custom coords and kwargs
    inset_connector(fig, ax1, ax2, coord1=coord1, coord2=coord2, **kwargs)

    # Check if ConnectionPatch is called once
    assert mock_connection_patch.call_count == 1

    # Verify that the keyword arguments were passed correctly
    call_args = mock_connection_patch.call_args
    assert call_args[1]["xyA"] == (1, 2)
    assert call_args[1]["xyB"] == (6, 7)
    assert call_args[1]["color"] == "red"
    assert call_args[1]["linestyle"] == "--"


@patch("m3util.viz.layout.ConnectionPatch", autospec=True)
def test_inset_connector_with_partial_coords(mock_connection_patch):
    """Test the function with coord1 provided and coord2 left as None."""
    # Create a figure and two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Custom coordinates for coord1 and None for coord2
    coord1 = [(1, 2)]

    # Set limits for ax2
    ax2.set_xlim(5, 15)
    ax2.set_ylim(10, 30)

    # Call the function with custom coord1 and default coord2
    inset_connector(fig, ax1, ax2, coord1=coord1)

    # Check if ConnectionPatch is called once
    assert mock_connection_patch.call_count == 1

    # Verify the arguments for the connection points
    call_args = mock_connection_patch.call_args
    p1 = call_args[1]["xyA"]
    p2 = call_args[1]["xyB"]

    # p1 should be the custom coord1[0] and p2 the bottom-left corner of ax2
    assert p1 == (1, 2)
    assert p2 == (5, 10)  # xlim[0], ylim[0] for ax2


@patch("m3util.viz.layout.path_maker")
@patch("matplotlib.axes.Axes.text")
def test_scalebar_tr_location(mock_text, mock_path_maker):
    """Test scalebar with 'tr' (top-right) location."""
    fig, ax = plt.subplots()

    # Set mock x and y limits for the axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Set the image size and scale size
    image_size = 100
    scale_size = 20  # nm

    # Call the scalebar function with 'tr' location
    scalebar(ax, image_size=image_size, scale_size=scale_size, loc="tr")

    # Check if path_maker was called with the correct arguments
    mock_path_maker.assert_called_once()
    path_args = mock_path_maker.call_args[0][
        1
    ]  # Get the path arguments (x_start, x_end, y_start, y_end)

    # Check that the scalebar coordinates are in the top-right
    x_start, x_end, y_start, y_end = path_args
    assert x_start > x_end  # In 'tr' x_start should be greater than x_end
    assert y_start > y_end  # In 'tr' y_start should be greater than y_end

    # Verify that the text label is placed correctly
    mock_text.assert_called_once()
    text_args = mock_text.call_args[0]
    text_x = text_args[0]
    label = text_args[2]

    # The text should be placed between x_start and x_end, and near the top-right
    assert x_start > x_end
    assert text_x == pytest.approx((x_start + x_end) / 2)
    assert label == "{0} {1}".format(scale_size, "nm")

    # Check that the correct stroke effect is applied
    path_effects = mock_text.call_args[1]["path_effects"]
    assert isinstance(path_effects[0], patheffects.withStroke)


@patch("m3util.viz.layout.scalebar")
def test_add_scalebar_with_valid_input(mock_scalebar):
    """Test that scalebar is called with correct arguments when scalebar_ is provided."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Example scalebar dictionary
    scalebar_dict = {"width": 100, "scale length": 10, "units": "nm"}

    # Call the add_scalebar function with the scalebar dictionary
    add_scalebar(ax, scalebar_dict)

    # Assert that the scalebar function was called once
    mock_scalebar.assert_called_once()

    # Check that the scalebar function was called with the correct arguments
    mock_scalebar.assert_called_with(ax, 100, 10, units="nm")


@patch("m3util.viz.layout.scalebar")
def test_add_scalebar_with_none(mock_scalebar):
    """Test that scalebar is not called when scalebar_ is None."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Call the add_scalebar function with None for scalebar_
    add_scalebar(ax, None)

    # Assert that the scalebar function was not called
    mock_scalebar.assert_not_called()


def test_layout_subfigures_with_margins():
    """Test that subfigures are correctly positioned with margins applied."""
    # Define the size of the overall figure (in inches)
    figure_size = (10, 8)

    # Define subfigure details with margins
    subfigures_dict = {
        "subfig1": {
            "position": (1, 1, 4, 4),  # (x, y, width, height) in inches
            "plot_func": MagicMock(),  # Mock the plot function
        },
        "subfig2": {
            "position": (6, 1, 4, 4),  # (x, y, width, height) in inches
            "plot_func": MagicMock(),  # Mock the plot function
        },
    }

    # Call the layout function with margin_pts = 72 (1 inch)
    fig, axes_dict = layout_subfigures_inches(
        figure_size, subfigures_dict, margin_pts=72
    )

    # Assert that the figure is created
    assert isinstance(fig, plt.Figure)

    # Check if the axes are created
    assert "subfig1" in axes_dict
    assert "subfig2" in axes_dict

    # Check that the axes positions respect the applied margins
    ax1_position = axes_dict["subfig1"].get_position()
    ax2_position = axes_dict["subfig2"].get_position()

    # Calculate expected positions based on margins
    # Margin is 1 inch (72 points)
    expected_ax1_left = (1 + 1) / 10
    expected_ax1_bottom = (1 + 1) / 8
    expected_ax1_width = (4 - 1) / 10
    expected_ax1_height = (4 - 1) / 8

    expected_ax2_left = (6 + 1) / 10
    expected_ax2_bottom = (1 + 1) / 8
    expected_ax2_width = (4 - 1) / 10
    expected_ax2_height = (4 - 1) / 8

    # Check the positions of the axes
    assert ax1_position.x0 == pytest.approx(expected_ax1_left)
    assert ax1_position.y0 == pytest.approx(expected_ax1_bottom)
    assert ax1_position.width == pytest.approx(expected_ax1_width)
    assert ax1_position.height == pytest.approx(expected_ax1_height)

    assert ax2_position.x0 == pytest.approx(expected_ax2_left)
    assert ax2_position.y0 == pytest.approx(expected_ax2_bottom)
    assert ax2_position.width == pytest.approx(expected_ax2_width)
    assert ax2_position.height == pytest.approx(expected_ax2_height)


def test_layout_subfigures_without_margins():
    """Test that subfigures are correctly positioned when margins are skipped."""
    # Define the size of the overall figure (in inches)
    figure_size = (10, 8)

    # Define subfigure details with no margins
    subfigures_dict = {
        "subfig1": {
            "position": (1, 1, 4, 4),  # (x, y, width, height) in inches
            "plot_func": MagicMock(),  # Mock the plot function
            "skip_margin": True,  # Skip margins
        },
        "subfig2": {
            "position": (6, 1, 4, 4),  # (x, y, width, height) in inches
            "plot_func": MagicMock(),  # Mock the plot function
            "skip_margin": True,  # Skip margins
        },
    }

    # Call the layout function with no margins
    fig, axes_dict = layout_subfigures_inches(
        figure_size, subfigures_dict, margin_pts=72
    )

    # Assert that the figure is created
    assert isinstance(fig, plt.Figure)

    # Check if the axes are created
    assert "subfig1" in axes_dict
    assert "subfig2" in axes_dict

    # Check that the axes positions respect skipping margins
    ax1_position = axes_dict["subfig1"].get_position()
    ax2_position = axes_dict["subfig2"].get_position()

    # Calculate expected positions with no margins
    expected_ax1_left = 1 / 10
    expected_ax1_bottom = 1 / 8
    expected_ax1_width = 4 / 10
    expected_ax1_height = 4 / 8

    expected_ax2_left = 6 / 10
    expected_ax2_bottom = 1 / 8
    expected_ax2_width = 4 / 10
    expected_ax2_height = 4 / 8

    # Check the positions of the axes
    assert ax1_position.x0 == pytest.approx(expected_ax1_left)
    assert ax1_position.y0 == pytest.approx(expected_ax1_bottom)
    assert ax1_position.width == pytest.approx(expected_ax1_width)
    assert ax1_position.height == pytest.approx(expected_ax1_height)

    assert ax2_position.x0 == pytest.approx(expected_ax2_left)
    assert ax2_position.y0 == pytest.approx(expected_ax2_bottom)
    assert ax2_position.width == pytest.approx(expected_ax2_width)
    assert ax2_position.height == pytest.approx(expected_ax2_height)


def test_layout_subfigures_with_right_margin():
    """Test that subfigures with 'right' key apply multiple margins."""
    # Define the size of the overall figure (in inches)
    figure_size = (10, 8)

    # Define subfigure details with the 'right' key set to True
    subfigures_dict = {
        "subfig1": {
            "position": (1, 1, 4, 4),  # (x, y, width, height) in inches
            "plot_func": MagicMock(),  # Mock the plot function
            "right": True,  # Apply double margin
        }
    }

    # Call the layout function with margin_pts = 72 (1 inch)
    fig, axes_dict = layout_subfigures_inches(
        figure_size, subfigures_dict, margin_pts=72
    )

    # Assert that the figure is created
    assert isinstance(fig, plt.Figure)

    # Check if the axes are created
    assert "subfig1" in axes_dict

    # Check that the axes positions respect the double margins
    ax1_position = axes_dict["subfig1"].get_position()

    # Calculate expected positions based on margins (multiple = 2)
    expected_ax1_left = (1 + 1) / 10
    expected_ax1_bottom = (1 + 1) / 8
    expected_ax1_width = (4 - 2 * 1) / 10  # Double margin applied to width
    expected_ax1_height = (4 - 2 * 1) / 8  # Double margin applied to height

    # Check the positions of the axes
    assert ax1_position.x0 == pytest.approx(expected_ax1_left)
    assert ax1_position.y0 == pytest.approx(expected_ax1_bottom)
    assert ax1_position.width == pytest.approx(expected_ax1_width)
    assert ax1_position.height == pytest.approx(expected_ax1_height)


def test_get_zorders_with_lines():
    """Test that the z-order of line objects in the figure is correctly retrieved."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Add a line to the axes with a specific z-order
    (line,) = ax.plot([0, 1], [0, 1], zorder=3)

    # Call the get_zorders function
    zorders = get_zorders(fig)

    # Assert that the line's z-order is in the zorders list
    assert any("Line2D" in desc and z == 3 for desc, z in zorders)


def test_get_zorders_with_text():
    """Test that the z-order of text objects in the figure is correctly retrieved."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Add text to the axes with a specific z-order
    ax.text(0.5, 0.5, "Test Text", zorder=4)

    # Call the get_zorders function
    zorders = get_zorders(fig)

    # Assert that the text's z-order is in the zorders list
    assert any("Text" in desc and z == 4 for desc, z in zorders)


def test_get_zorders_with_ticks():
    """Test that the z-order of tick labels is correctly retrieved."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Set some tick labels
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.set_xticklabels(["A", "B", "C"])

    # Call the get_zorders function
    zorders = get_zorders(fig)

    # Assert that tick labels' z-orders are in the zorders list
    assert any("Tick Label (A)" in desc for desc, z in zorders)
    assert any("Tick Label (B)" in desc for desc, z in zorders)
    assert any("Tick Label (C)" in desc for desc, z in zorders)


def test_get_zorders_with_multiple_axes():
    """Test that z-orders are correctly retrieved for figures with multiple axes."""
    # Create a figure with multiple axes
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Add elements to both axes
    ax1.plot([0, 1], [0, 1], zorder=2)
    ax2.text(0.5, 0.5, "Test on Ax2", zorder=5)

    # Call the get_zorders function
    zorders = get_zorders(fig)

    # Assert that z-orders from both axes are in the zorders list
    assert any("Line2D" in desc and z == 2 for desc, z in zorders)
    assert any("Text" in desc and z == 5 for desc, z in zorders)


def test_get_zorders_with_no_zorder_items():
    """Test that the function handles cases where some items don't have z-order."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Add a patch that doesn't have a z-order attribute
    ax.add_patch(plt.Circle((0.5, 0.5), radius=0.1))  # Default zorder

    # Call the get_zorders function
    zorders = get_zorders(fig)

    # Assert that no errors occurred and the result is still valid
    assert isinstance(zorders, list)


def test_get_zorders_with_no_axes():
    """Test that the function handles figures with no axes."""
    # Create a figure with no axes
    fig = plt.figure()

    # Call the get_zorders function
    zorders = get_zorders(fig)

    # Assert that the zorders list is empty
    assert len(zorders) == 0


@patch(
    "m3util.viz.layout.line_annotation"
)  # Mock line_annotation since it's used in the function
def test_draw_line_with_text_span_data_vertical_(mock_line_annotation):
    """Test for drawing a vertical line between closest y-values for span='data'."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Define x_data and y_data
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 15, 20, 25, 30])

    # Define the value for the line
    value = 3.5  # Falls between x_data values

    # Call the function with span="data" and axis="x"
    draw_line_with_text(
        ax, x_data, y_data, value, axis="x", span="data", text="Test Line"
    )

    # Check that the line was drawn between the closest y-values (20 and 25)
    lines = ax.get_lines()
    assert len(lines) == 1  # One line should be drawn

    line_x = lines[0].get_xdata()
    line_y = lines[0].get_ydata()

    # Verify that the vertical line is at x=3.5 and spans the correct y-values
    assert np.allclose(line_x, [3.5, 3.5])
    assert np.allclose(line_y, [20, 25])

    # Verify that the line_annotation function was called for the text
    mock_line_annotation.assert_called_once()


@patch(
    "m3util.viz.layout.line_annotation"
)  # Mock line_annotation since it's used in the function
def test_draw_line_with_text_span_data_horizontal(mock_line_annotation):
    """Test for drawing a horizontal line between closest x-values for span='data'."""
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Define x_data and y_data
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([10, 15, 20, 25, 30])

    # Define the value for the line
    value = 22.5  # Falls between y_data values

    # Call the function with span="data" and axis="y"
    draw_line_with_text(
        ax, x_data, y_data, value, axis="y", span="data", text="Test Line"
    )

    # Check that the line was drawn between the closest x-values (3 and 4)
    lines = ax.get_lines()
    assert len(lines) == 1  # One line should be drawn

    line_x = lines[0].get_xdata()
    line_y = lines[0].get_ydata()

    # Verify that the horizontal line is at y=22.5 and spans the correct x-values
    assert np.allclose(line_y, [22.5, 22.5])
    assert np.allclose(line_x, [3, 4])

    # Verify that the line_annotation function was called for the text
    mock_line_annotation.assert_called_once()


def test_plot_into_graph_with_clim(sample_figure):
    fig, ax = sample_figure
    fig_test, ax_test = plt.subplots()
    clim = (0, 1)
    plot_into_graph(ax, fig_test, clim=clim)
    assert len(ax.get_images()) > 0, "The image should be plotted into the axes."
    assert (
        ax.get_images()[0].get_clim() == clim
    ), "The color limits should be set correctly."


def test_plot_into_graph_with_colorbar(sample_figure):
    fig, ax = sample_figure
    fig_test, ax_test = plt.subplots()
    plot_into_graph(ax, fig_test, colorbar_=True)
    assert len(ax.get_images()) > 0, "The image should be plotted into the axes."


def test_layout_fig_mod_2():
    fig, axes = layout_fig(2)
    assert len(axes) == 2, "Should create 2 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert axes.shape == (2,), "Should return a 1D array of axes."


def test_layout_fig_mod_3():
    fig, axes = layout_fig(4)
    assert len(axes) == 4, "Should create 4 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert axes.shape == (4,), "Should return a 1D array of axes."


def test_layout_fig_mod_4():
    fig, axes = layout_fig(9)
    assert len(axes) == 9, "Should create 9 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert axes.shape == (9,), "Should return a 1D array of axes."


def test_layout_fig_mod_5():
    fig, axes = layout_fig(16)
    assert len(axes) == 16, "Should create 16 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert axes.shape == (16,), "Should return a 1D array of axes."


def test_layout_fig_mod_6():
    fig, axes = layout_fig(25)
    assert len(axes) == 25, "Should create 25 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert axes.shape == (25,), "Should return a 1D array of axes."


def test_layout_fig_mod_7():
    fig, axes = layout_fig(36)
    assert len(axes) == 36, "Should create 36 subplots."
    assert isinstance(fig, plt.Figure), "Should return a matplotlib figure."
    assert axes.shape == (36,), "Should return a 1D array of axes."
