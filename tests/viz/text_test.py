from m3util.viz.text import add_text_to_figure
import matplotlib.pyplot as plt
import pytest


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
