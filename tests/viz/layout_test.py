import pytest
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, ConnectionPatch
from itertools import product
from m3util.viz.layout import (
    plot_into_graph,
    subfigures,
    add_text_to_figure,
    add_box,
    inset_connector,
    path_maker,
    layout_fig,
    embedding_maps,
    imagemap,
    find_nearest,
    combine_lines,
    labelfigs,
    number_to_letters,
    scalebar,
    Axis_Ratio,
    get_axis_range,
    set_axis,
    add_scalebar,
    get_axis_pos_inches,
    FigDimConverter
)

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
    assert any(isinstance(patch, Rectangle) for patch in ax.patches), "A rectangle should be added to the axes."

def test_inset_connector(sample_figure):
    fig, ax = sample_figure
    fig2, ax2 = plt.subplots()
    inset_connector(fig, ax, ax2)
    assert len(fig.findobj(ConnectionPatch)) > 0, "A connection patch should be added between axes."

def test_path_maker(sample_figure):
    fig, ax = sample_figure
    path_maker(ax, (0.1, 0.4, 0.1, 0.4), 'blue', 'red', '-', 2)
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
    assert number_to_letters(0) == 'a', "Number to letter conversion should return 'a' for 0."
    assert number_to_letters(25) == 'z', "Number to letter conversion should return 'z' for 25."
    assert number_to_letters(26) == 'aa', "Number to letter conversion should return 'aa' for 26."

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
    add_scalebar(ax, {'width': 100, 'scale length': 10, 'units': 'nm'})
    assert len(ax.patches) > 0, "A scalebar should be added to the axes."

def test_get_axis_pos_inches(sample_figure):
    fig, ax = sample_figure
    pos_inches = get_axis_pos_inches(fig, ax)
    assert isinstance(pos_inches, np.ndarray), "Position should be returned as a numpy array."

def test_FigDimConverter():
    converter = FigDimConverter((10, 5))
    inches = converter.to_inches((0.5, 0.5, 0.5, 0.5))
    assert inches == (5, 2.5, 5, 2.5), "Should convert relative dimensions to inches correctly."
    relative = converter.to_relative((5, 2.5, 5, 2.5))
    assert relative == (0.5, 0.5, 0.5, 0.5), "Should convert inches to relative dimensions correctly."

def test_add_box():
    fig, ax = plt.subplots()
    
    # Define the position of the box
    pos = (0.1, 0.2, 0.4, 0.6)
    
    # Call the add_box function
    add_box(ax, pos, edgecolor='red', facecolor='none', linewidth=2)
    
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
    assert rect.get_width() == pos[2] - pos[0], "The width of the rectangle is incorrect"
    assert rect.get_height() == pos[3] - pos[1], "The height of the rectangle is incorrect"
    
    # Check that the rectangle has the correct properties
    assert rect.get_edgecolor() == (1.0, 0.0, 0.0, 1.0), "The edge color of the rectangle is incorrect"
    assert rect.get_facecolor() == (0.0, 0.0, 0.0, 0.0), "The face color of the rectangle is incorrect"
    assert rect.get_linewidth() == 2, "The line width of the rectangle is incorrect"
    
def test_inset_connector():
    # Create a figure and two axes
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Set limits for both axes to test connection points calculation
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 20)
    
    # Call the inset_connector function
    inset_connector(fig, ax1, ax2, color='blue', linestyle='--', linewidth=2)
    
    # Retrieve the added ConnectionPatch objects from the figure
    connections = [obj for obj in fig.findobj(ConnectionPatch)]
    
    # Check that exactly two connections have been added (since coord1 and coord2 each have 2 points by default)
    assert len(connections) == 2, "Expected exactly two ConnectionPatch objects to be added to the figure."
    
    # Check the properties of the connections
    for con in connections:
        assert con.get_edgecolor()[:3] == (0.0, 0.0, 1.0), "The color of the ConnectionPatch should be blue."
        assert con.get_linestyle() == '--', "The linestyle of the ConnectionPatch should be dashed."
        assert con.get_linewidth() == 2, "The linewidth of the ConnectionPatch should be 2."

    # Test with specific coordinates
    coord1 = [(1, 1), (1, 9)]
    coord2 = [(2, 2), (2, 18)]
    
    # Call the inset_connector function with specific coordinates
    inset_connector(fig, ax1, ax2, coord1=coord1, coord2=coord2, color='green', linestyle='-', linewidth=1)
    
    # Retrieve the added ConnectionPatch objects again
    connections = [obj for obj in fig.findobj(ConnectionPatch)]
    
    # There should now be 4 ConnectionPatch objects in total
    assert len(connections) == 4, "Expected four ConnectionPatch objects to be added to the figure after the second call."
    
    # Check the properties of the last two connections with a tolerance for color comparison
    for con in connections[2:]:
        assert all(abs(a - b) < 0.01 for a, b in zip(con.get_edgecolor()[:3], (0.0, 0.5, 0.0))), "The color of the ConnectionPatch should be green."
        assert con.get_linestyle() == '-', "The linestyle of the ConnectionPatch should be solid."
        assert con.get_linewidth() == 1, "The linewidth of the ConnectionPatch should be 1."

def test_subfigures_default():
    # Test with default size and gaps
    nrows, ncols = 2, 3
    fig, ax = subfigures(nrows, ncols)
    
    # Check if the number of axes is correct
    assert len(ax) == nrows * ncols, f"Expected {nrows * ncols} subfigures, but got {len(ax)}."
    
    # Check the size of the figure
    expected_figsize = (1.25*ncols + 0.8*ncols, 1.25*nrows + 0.33*nrows)
    assert fig.get_size_inches() == pytest.approx(expected_figsize), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."

def test_subfigures_custom_size_and_gaps():
    # Test with custom size and gaps
    nrows, ncols = 3, 2
    custom_size = (2.0, 2.0)
    custom_gaps = (1.0, 0.5)
    fig, ax = subfigures(nrows, ncols, size=custom_size, gaps=custom_gaps)
    
    # Check if the number of axes is correct
    assert len(ax) == nrows * ncols, f"Expected {nrows * ncols} subfigures, but got {len(ax)}."
    
    # Check the size of the figure
    expected_figsize = (custom_size[0]*ncols + custom_gaps[0]*ncols, custom_size[1]*nrows + custom_gaps[1]*nrows)
    assert fig.get_size_inches() == pytest.approx(expected_figsize), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."

def test_subfigures_custom_figsize():
    # Test with a custom figsize
    nrows, ncols = 2, 2
    custom_figsize = (8, 6)
    fig, ax = subfigures(nrows, ncols, figsize=custom_figsize)
    
    # Check if the number of axes is correct
    assert len(ax) == nrows * ncols, f"Expected {nrows * ncols} subfigures, but got {len(ax)}."
    
    # Check the size of the figure
    assert np.allclose(fig.get_size_inches(), custom_figsize), f"Expected figure size {custom_figsize}, but got {fig.get_size_inches()}."


def test_subfigures_size():
    # Test with custom sizes
    nrows, ncols = 2, 3
    custom_size = (2.0, 2.0)
    fig, ax = subfigures(nrows, ncols, size=custom_size)
    
    # Check if the number of axes is correct
    assert len(ax) == nrows * ncols, f"Expected {nrows * ncols} subfigures, but got {len(ax)}."
    
    # Check the size of the figure
    expected_figsize = (custom_size[0]*ncols + 0.8*ncols, custom_size[1]*nrows + 0.33*nrows)
    assert np.allclose(fig.get_size_inches(), expected_figsize), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."

def test_subfigures_gaps():
    # Test with custom gaps
    nrows, ncols = 3, 2
    custom_gaps = (1.0, 0.5)
    fig, ax = subfigures(nrows, ncols, gaps=custom_gaps)
    
    # Check if the number of axes is correct
    assert len(ax) == nrows * ncols, f"Expected {nrows * ncols} subfigures, but got {len(ax)}."
    
    # Check the size of the figure
    expected_figsize = (1.25*ncols + custom_gaps[0]*ncols, 1.25*nrows + custom_gaps[1]*nrows)
    assert np.allclose(fig.get_size_inches(), expected_figsize), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."

def test_subfigures_size_and_gaps():
    # Test with custom sizes and gaps
    nrows, ncols = 2, 2
    custom_size = (2.5, 2.0)
    custom_gaps = (0.5, 1.0)
    fig, ax = subfigures(nrows, ncols, size=custom_size, gaps=custom_gaps)
    
    # Check if the number of axes is correct
    assert len(ax) == nrows * ncols, f"Expected {nrows * ncols} subfigures, but got {len(ax)}."
    
    # Check the size of the figure
    expected_figsize = (custom_size[0]*ncols + custom_gaps[0]*ncols, custom_size[1]*nrows + custom_gaps[1]*nrows)
    assert np.allclose(fig.get_size_inches(), expected_figsize), f"Expected figure size {expected_figsize}, but got {fig.get_size_inches()}."

def test_add_text_to_figure():
    # Create a figure
    fig = plt.figure(figsize=(8, 6))  # 8 inches by 6 inches
    
    # Add text at a specific position in inches
    text = "Sample Text"
    text_position_in_inches = (4, 3)  # Center of the figure
    kwargs = {'fontsize': 12, 'color': 'blue'}
    
    # Call the function to add the text
    add_text_to_figure(fig, text, text_position_in_inches, **kwargs)
    
    # Verify that the text was added correctly
    assert len(fig.texts) == 1, "Expected exactly one text element to be added to the figure."
    
    # Check the position of the text
    text_obj = fig.texts[0]
    expected_position = (text_position_in_inches[0] / fig.get_size_inches()[0],
                         text_position_in_inches[1] / fig.get_size_inches()[1])
    
    assert text_obj.get_position() == pytest.approx(expected_position), \
        f"Expected text position {expected_position}, but got {text_obj.get_position()}."
    
    # Check the text content
    assert text_obj.get_text() == text, f"Expected text content '{text}', but got '{text_obj.get_text()}'."
    
    # Check additional kwargs
    assert text_obj.get_fontsize() == kwargs['fontsize'], \
        f"Expected fontsize {kwargs['fontsize']}, but got {text_obj.get_fontsize()}."
    assert text_obj.get_color() == kwargs['color'], \
        f"Expected color {kwargs['color']}, but got {text_obj.get_color()}."

def test_add_text_to_figure_default():
    # Create a figure with default settings
    fig = plt.figure()
    
    # Add text at a specific position in inches
    text = "Default Position Text"
    text_position_in_inches = (2, 1)  # Arbitrary position
    add_text_to_figure(fig, text, text_position_in_inches)
    
    # Verify that the text was added correctly
    assert len(fig.texts) == 1, "Expected exactly one text element to be added to the figure."
    
    # Check the position of the text
    text_obj = fig.texts[0]
    expected_position = (text_position_in_inches[0] / fig.get_size_inches()[0],
                         text_position_in_inches[1] / fig.get_size_inches()[1])
    
    assert text_obj.get_position() == pytest.approx(expected_position), \
        f"Expected text position {expected_position}, but got {text_obj.get_position()}."
    
    # Check the text content
    assert text_obj.get_text() == text, f"Expected text content '{text}', but got '{text_obj.get_text()}'."